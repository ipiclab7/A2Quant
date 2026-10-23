"""Native FP32 ACM control, sharing data/loss/validation with the quantized runs."""
import hashlib
import json
from pathlib import Path
import random
import tarfile
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import SirstDataset, seed_worker
from .models import ASKCResUNet
from .objectives import soft_iou_loss
from .quantization import named_quantizers, bit_statistics
from .model_cost import model_cost
from .splits import load_verified_splits
from .training import atomic_save, source_hash, rng_state, restore_rng, update_learning_rates, validate
from .stability import bn_digest


def build_fp_model(config):
    model = ASKCResUNet(**config['model'])
    if named_quantizers(model):
        raise RuntimeError('FP baseline must contain no quantizers')
    return model


def run(config, root, output, device='cuda:0', smoke=False, stop_after=None, resume=False):
    root, output = Path(root), Path(output)
    if not torch.cuda.is_available() or not device.startswith('cuda'):
        raise RuntimeError('CUDA is required for this recorded baseline')
    if (config['epochs'], config['image_size'], config['batch_size']) != (1500, 512, 16):
        raise ValueError('FP control preserves 1500 epochs, 512 input, batch16')
    if config['preq_epochs'] != 0 or config['bn_policy'] != 'standard' or config['experiment_kind'] != 'fp_baseline':
        raise ValueError('FP baseline has no PreQ/AdaQAT and uses standard BN')
    if config['dtype'] != 'float32' or config['amp'] or config['allow_tf32']:
        raise ValueError('FP32 baseline records AMP/TF32 disabled')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed = config['training_seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dataset_config, splits = load_verified_splits(config['data_root'], root/config['dataset_config'])
    provenance = json.loads((Path(config['data_root'])/'provenance.json').read_text())
    identity = {'config': config, 'dataset_config': dataset_config,
                'split_hashes': provenance['split_file_sha256'], 'code_sha256': source_hash(root), 'smoke': smoke}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    total, max_batches = (2, 2) if smoke else (1500, None)
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError('Nonempty run; use --resume or a new run identity')
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    workers = 0 if smoke else config['num_workers']
    train_loader = DataLoader(SirstDataset(config['data_root'], splits['train'], True),
        batch_size=16, shuffle=True, num_workers=workers, pin_memory=True,
        generator=generator, worker_init_fn=seed_worker)
    val_loader = DataLoader(SirstDataset(config['data_root'], splits['val']),
        batch_size=16, shuffle=False, num_workers=workers, pin_memory=True, worker_init_fn=seed_worker)
    model = build_fp_model(config).to(device)
    if config['network_optimizer'] != 'Adagrad':
        raise ValueError('The control must use the recorded network Adagrad')
    optimizer = torch.optim.Adagrad([{'params': list(model.parameters()), 'lr': config['network_lr'],
                                    'weight_decay': config['weight_decay'], 'name': 'network'}])
    start, best_iou, best_epoch, elapsed_previous = 1, -1., None, 0.
    if resume:
        saved = torch.load(output/'last.pt', map_location='cpu')
        if saved['signature'] != signature:
            raise ValueError('FP resume config/source/splits differ')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start, best_iou, best_epoch = saved['epoch']+1, saved['best_iou'], saved['best_epoch']
        elapsed_previous = saved['elapsed_training_seconds']
        restore_rng(saved['rng'], generator)
        if (output/'final-test.json').exists():
            if json.loads((output/'final-test.json').read_text())['signature'] != signature:
                raise ValueError('Existing test signature differs')
            print('FP RUN ALREADY COMPLETE; existing test retained', flush=True)
            return
    else:
        (output/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        with tarfile.open(output/'source-snapshot.tar.gz','w:gz') as archive:
            files=list((root/'src/a2quant').rglob('*.py'))+list((root/'configs').glob('*.json'))+[root/'scripts/train.py',root/'LICENSE']
            for path in sorted(files):archive.add(path,arcname=str(path.relative_to(root)))
    metadata=dict(identity, signature=signature, result_type='smoke_only' if smoke else 'native_fp32_acm_control_unverified_author_configuration',
                  effective_total_epochs=total, precision_mode='fp32', effective_preq_epochs=0,
                  torch_version=torch.__version__, cuda_build=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), test_accessed=False, checkpoint_selection='maximum validation IoU across FP training')
    (output/'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    end=min(total,stop_after) if stop_after is not None else total
    for epoch in range(start,end+1):
        tick=time.monotonic(); model.train(); update_learning_rates(optimizer,config,epoch)
        count, loss_sum=0,0.
        for index,(images,targets,_) in enumerate(train_loader):
            if max_batches is not None and index>=max_batches:break
            images,targets=images.to(device,non_blocking=True),targets.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss=soft_iou_loss(model(images),targets)
            if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite FP training loss')
            loss.backward()
            gradients=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not torch.stack(gradients).all().item():raise RuntimeError('Nonfinite FP training gradient')
            optimizer.step(); count+=len(images);loss_sum+=loss.item()*len(images)
        bn_hash=bn_digest(model)
        validation=validate(model,val_loader,device,max_batches)
        improved=validation['IoU']>best_iou
        if improved:best_iou,best_epoch=validation['IoU'],epoch
        row={'epoch':epoch,'stage':'fp32','train_samples':count,
             'train':{'task':loss_sum/count,'regularization':0.,'total':loss_sum/count},
             'validation':validation,'bits':bit_statistics(model),
             'learning_rates':{g['name']:g['lr'] for g in optimizer.param_groups},
             'bn':{'policy':'standard','frozen':False,'buffer_sha256':bn_hash},
             'quantizers_count':len(named_quantizers(model)),
             'epoch_seconds':time.monotonic()-tick,'cuda_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
             'best_fp_epoch':best_epoch}
        saved={'epoch':epoch,'stage':'fp32','model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'rng':rng_state(generator),'signature':signature,'config':config,'best_iou':best_iou,
               'best_epoch':best_epoch,'elapsed_training_seconds':elapsed_previous+time.monotonic()-started}
        atomic_save(saved,output/'last.pt')
        if improved:atomic_save(saved,output/'best_fp32.pt')
        with (output/'epochs.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    if end<total:
        print('STOPPED_AT_EPOCH_LIMIT; FP resume supported; test untouched',flush=True);return
    if smoke:
        (output/'smoke-complete.json').write_text(json.dumps({'passed':True,'test_accessed':False,'precision_mode':'fp32'}))
        print('FP SMOKE COMPLETE; not a paper result; test untouched',flush=True);return
    best=torch.load(output/'best_fp32.pt',map_location='cpu');model.load_state_dict(best['model'])
    costs=model_cost(model,torch.zeros(1,3,512,512,device=device));costs['actual_checkpoint_bytes']=(output/'best_fp32.pt').stat().st_size
    (output/'final-model-cost.json').write_text(json.dumps(costs,indent=2)+'\n',encoding='utf8')
    test_loader=DataLoader(SirstDataset(config['data_root'],splits['test']),batch_size=16,shuffle=False,
                          num_workers=workers,pin_memory=True,worker_init_fn=seed_worker)
    test=validate(model,test_loader,device,objects=True)
    test.update(selected_epoch=best['epoch'],validation_IoU_at_selection=best['best_iou'],
                split_sha256=provenance['split_file_sha256']['test'],bits=bit_statistics(model),
                authors_exact_configuration_verified=False,signature=signature,
                elapsed_training_seconds=elapsed_previous+time.monotonic()-started,result_type=metadata['result_type'])
    (output/'final-test.json').write_text(json.dumps(test,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    metadata.update(test_accessed=True,training_complete=True)
    (output/'run.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('FP TRAINING AND FINAL TEST COMPLETE',json.dumps(test),flush=True)
