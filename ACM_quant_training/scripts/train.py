"""Portable entry for the saved-LR FP104 quantization method; no automatic TEST."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R), str(R / 'src'), str(R / 'scripts')]
sys.dont_write_bytecode = True
from configuration import get_training_config, DATASET_CONFIG, DATASET_PROVENANCE_SHA256, REQUIRED_PACKAGES

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(p)

def check_package():
    from source_manifest import MANIFEST
    for relative, expected in MANIFEST.items():
        path = (R / relative).resolve()
        path.relative_to(R)
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('Source file changed: ' + relative)
    return len(MANIFEST)

def load_verified_splits(data_root, unused_config_path=None):
    root = Path(data_root)
    provenance = read(root / 'provenance.json')
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != DATASET_PROVENANCE_SHA256:
        raise ValueError('External dataset provenance differs from the frozen source')
    splits = {}
    for role, expected_count in [('train', 213), ('val', 128), ('test', 86)]:
        p = root / 'splits' / DATASET_CONFIG['split_group'] / (role + '.txt')
        if sha(p) != provenance['split_file_sha256'][role]:
            raise ValueError('Frozen split hash mismatch: ' + role)
        ids = p.read_text().splitlines()
        if len(ids) != expected_count or len(set(ids)) != expected_count:
            raise ValueError('Invalid split count: ' + role)
        if any(not name or Path(name).name != name or '\\' in name or name in {'.', '..'} for name in ids):
            raise ValueError('Invalid sample ID')
        splits[role] = ids
    all_ids = [name for ids in splits.values() for name in ids]
    if len(set(all_ids)) != 427 or set(all_ids) != {row['id'] for row in provenance['samples_manifest']}:
        raise ValueError('Split overlap or incomplete coverage')
    for name in all_ids:
        if not (root / 'images' / (name + '.png')).is_file() or not (root / 'masks' / (name + '.png')).is_file():
            raise FileNotFoundError('Incomplete image/mask pair: ' + name)
    return dict(DATASET_CONFIG), splits


def check_dataset(data_root, required):
    root = Path(data_root)
    if not root.exists():
        if required:
            raise FileNotFoundError('Supply the frozen SIRST data using --data-root')
        return 'external_dataset_required'
    _, splits = load_verified_splits(root)
    allowed = set(splits['train'] + splits['val'])
    for row in read(root / 'provenance.json')['samples_manifest']:
        if row['id'] in allowed:
            for field, directory in [('image_sha256', 'images'), ('mask_sha256', 'masks')]:
                if sha(root / directory / (row['id'] + '.png')) != row[field]:
                    raise ValueError('TRAIN/VAL file hash mismatch: ' + row['id'])
    return 'frozen_split_verified_TEST_images_unread'

def load_runtime():
    global torch, run, bridge, digest, SquareWarpDataset, load_source_components
    global named_quantizers, bn_digest, copy
    import copy
    import torch
    import acm_quant_runner as run
    import pytorch_quant_bridge as bridge
    from state_digest import digest
    from pytorch_geometry_data import SquareWarpDataset
    from upstream_pytorch_bridge import load_source_components
    from a2quant.quantization import named_quantizers
    from a2quant.stability import bn_digest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='Stdlib package/data checks only; no torch or model')
    mode.add_argument('--train', action='store_true', help='Explicitly start a NEW bounded1500-epoch run')
    mode.add_argument('--print-requirements', action='store_true', help='Print required pip packages without writing inside the example')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--parent-checkpoint', type=Path,
                        help='External FP104 checkpoint containing model and full Adagrad state; never bundled')
    parser.add_argument('--activation-ranges', type=Path, help='External frozen TRAIN activation-range JSON')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    count = check_package()
    if args.print_requirements:
        print('\n'.join(REQUIRED_PACKAGES))
        return
    c = get_training_config()
    if args.data_root is not None:
        c['data_root'] = str(args.data_root.resolve())
    parent_status = 'external_parent_checkpoint_required'
    if args.parent_checkpoint is not None:
        parent = args.parent_checkpoint.resolve()
        if parent == R or R in parent.parents:
            raise ValueError('Keep weights outside the code package')
        if not parent.is_file():
            raise FileNotFoundError('External parent checkpoint does not exist: ' + str(parent))
        if sha(parent) != c['parent_checkpoint_sha256']:
            raise ValueError('Expected the recorded FP104 checkpoint with full Adagrad state')
        c['parent_checkpoint'] = str(parent)
        parent_status = 'external_FP104_SHA_verified'
    elif args.train:
        parser.error('--train requires --parent-checkpoint pointing to the external FP104 checkpoint')
    ranges_status = 'external_activation_ranges_required'
    if args.activation_ranges is not None:
        ranges = args.activation_ranges.resolve()
        if ranges == R or R in ranges.parents:
            raise ValueError('Keep calibration inputs outside the source example')
        if not ranges.is_file() or sha(ranges) != c['activation_ranges_sha256']:
            raise ValueError('Expected the recorded frozen TRAIN activation-range JSON')
        c['activation_ranges_path'] = str(ranges)
        ranges_status = 'external_activation_ranges_SHA_verified'
    elif args.train:
        parser.error('--train requires --activation-ranges pointing to the external frozen TRAIN ranges')
    data_status = check_dataset(c['data_root'], required=args.train)
    if args.check:
        print(json.dumps(dict(package_files_verified=count, dataset=data_status, parent_checkpoint=parent_status, activation_ranges=ranges_status,
                             model_loads=0, forwards=0, TRAIN=0, VAL=0, TEST=0)))
        return
    if args.output_dir is None:
        parser.error('--train requires --output-dir pointing to a new run directory')
    out = args.output_dir.resolve()
    # Keep output separate from the delivered package and immutable inputs.
    if out == R or R in out.parents:
        raise ValueError('Choose an output directory outside this immutable code package')
    if args.resume and ((out / 'quality-stop.json').exists() or (out / 'trainval-complete.json').exists()):
        raise ValueError('A quality-stopped or completed run cannot be resumed or extended')
    load_runtime()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES')
    import subprocess
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if any(row.strip().isdigit() and int(row.strip()) != os.getpid() for row in gpu_rows.splitlines()):
        raise RuntimeError('GPU compute jobs already exist; wait, do not kill them')
    _, splits = load_verified_splits(c['data_root'], R / c['dataset_config'])
    from PIL import Image
    allowed = set(splits['train'] + splits['val'])
    original_open = Image.open
    def guarded_open(path, *a, **k):
        p = Path(path).resolve()
        root = Path(c['data_root']).resolve()
        if p.parent not in [root / 'images', root / 'masks'] or p.stem not in allowed:
            raise ValueError('TEST/unknown image access rejected')
        return original_open(path, *a, **k)
    Image.open = guarded_open
    install(c, out, splits)
    run.run(c, R, out, resume=args.resume)

def raw(c,device='cpu'):
 cls,_,_,_=load_source_components(R);m=cls(**c['model']);v=torch.load(R/c['parent_checkpoint'],map_location='cpu')
 assert v['epoch']==104 and sha(R/c['parent_checkpoint'])==c['parent_checkpoint_sha256'];m.load_state_dict(v['model'],strict=True)
 return m.to(device)


def policy(m,epoch=0,boundary=0,policy=None):
 m.train()
 for b in m.modules():
  if isinstance(b,torch.nn.modules.batchnorm._BatchNorm):b.eval()
 b=[x for x in m.modules() if isinstance(x,torch.nn.modules.batchnorm._BatchNorm)]
 assert len(b)==59 and all(int(x.num_batches_tracked)==18452 and x.weight.requires_grad and x.bias.requires_grad for x in b)
 assert all(x.training for x in m.modules() if isinstance(x,torch.nn.modules.dropout._DropoutNd))
 return True


def build(c):
 m=raw(c);parent=digest(m.state_dict());parent_bn=bn_digest(m)
 bridge.convert_source(m,**c['quantization'])
 ranges=read(R/c['activation_ranges_path']);assert sha(R/c['activation_ranges_path'])==c['activation_ranges_sha256']
 with torch.no_grad():
  for name,q in named_quantizers(m):
   if not q.signed:
    value=max(float(ranges[name]),1e-6);assert value<=1e4;q.alpha.fill_(value)
 q=named_quantizers(m);ids={id(p) for _,v in q for p in [v.alpha,v.beta]}
 assert len(q)==113 and sum(v.signed for _,v in q)==61 and sum(p.numel() for p in m.parameters() if id(p) not in ids)==519843
 physical={n:v for n,v in m.state_dict().items() if '.weight_quantizer.' not in n and '.quantizer.' not in n and '.output_quantizer.' not in n}
 assert digest(physical)==parent and bn_digest(m)==parent_bn
 write(OUT/'warmstart.json',dict(parent_checkpoint=c['parent_checkpoint'],parent_sha256=c['parent_checkpoint_sha256'],parent_epoch=104,
  network_exact=True,network_sha256=parent,BN_sha256=parent_bn,W=61,A=52,learned_quantization_parameters=226,parent_network_Adagrad_state_loaded=True,fresh_quantization_Adam=True,FP_optimizer_not_loaded=False,activation_ranges_sha256=c['activation_ranges_sha256']))
 return m


def install(c,out,s):
 global OUT
 OUT=out
 # Base bridge only patches functions; its geometry scratch-only launcher is never called.
 bridge.configure_training(c,R,out)
 import a2quant.training as original
 for n in ['constrain_quantizers','stage_consistency','range_health','update_learning_rates']:setattr(run,n,getattr(original,n))
 def lr(opt,cfg,epoch):
  full=dict(cfg);full['epochs']=1500;original.update_learning_rates(opt,full,epoch)
 run.update_learning_rates=lr
 run.build_model=build;run.training_mode=policy
 run.load_verified_splits=load_verified_splits
 original_optimizer_builder=run.build_optimizer
 def optimizer_builder(model,cfg):
  opt=original_optimizer_builder(model,cfg);saved=torch.load(R/cfg['parent_checkpoint'],map_location='cpu')['optimizer']
  assert saved['param_groups'][0]['lr']==cfg['network_lr'],'Parent checkpoint LR must match registered base without scaling'
  opt.network.load_state_dict(saved)
  for group in opt.network.param_groups:group['lr']=cfg['network_lr']
  expected=copy.deepcopy(saved)
  for group in expected['param_groups']:group['lr']=cfg['network_lr']
  assert digest(opt.network.state_dict())==digest(expected)
  assert not opt.quantization.state_dict()['state']
  write(OUT/'network-optimizer-parent.json',dict(parent_checkpoint=cfg['parent_checkpoint'],parent_optimizer_digest=digest(saved),loaded_network_digest=digest(expected),all_accumulators_steps_and_WD_exact=True,parent_checkpoint_LR_preserved=True,initial_Q_network_lr=cfg['network_lr'],registered_parent_base_poly_schedule=True,fresh_quantization_Adam=True))
  return opt
 run.build_optimizer=optimizer_builder
 class Guard(SquareWarpDataset):
  def __init__(self,root,ids,training=False,size=512):
   assert list(ids)==s['train'] if training else list(ids) in [s['val'],s['train'][:4]]
   super().__init__(root,ids,training,size)
 run.SirstDataset=Guard

if __name__ == '__main__':
    main()
