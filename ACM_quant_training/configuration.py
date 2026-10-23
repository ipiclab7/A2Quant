"""Training parameters, integrity constants and display formula for the ACM example."""
from copy import deepcopy

TRAINING = {'model': {'layer_blocks': [4, 4, 4], 'channels': [8, 16, 32, 64], 'fuse_mode': 'AsymBi'},
 'image_size': 512,
 'batch_size': 16,
 'epochs': 1500,
 'preq_epochs': 300,
 'network_lr': 0.0007154879910294983,
 'quantization_lr': 0.001,
 'lambda_encoder': 0.5,
 'lambda_decoder': 0.5,
 'network_optimizer': 'Adagrad',
 'quantization_optimizer': 'Adam',
 'weight_decay': 0.0001,
 'schedule': {'warmup_epochs': 0,
              'power': 0.9,
              'min_lr_factor': 0.0001,
              'timing': 'parent source end-of-epoch before next; quantizer uses same relative '
                        'factor'},
 'quantization': {'weight_bit': 8.0, 'activation_bit': 8.0, 'min_bit': 2, 'max_bit': 14},
 'targets': {'encoder': {'weight': 4.0, 'activation': 5.0},
             'decoder': {'weight': 4.0, 'activation': 5.0}},
 'training_seed': 20261021,
 'num_workers': 0,
 'dtype': 'float32',
 'amp': False,
 'allow_tf32': False,
 'bn_policy': 'parent_fixed_all_stages',
 'range_consistency_checks': True,
 'weight_alpha_policy': 'current_absmax_upper_envelope',
 'parent_VAL': {'IoU': 0.7259694274116936,
                'nIoU': 0.698266666342332,
                'Pd': 0.9354838709677419,
                'Fa_per_million_pixels': 14.901161193847656},
 'experiment': 'portable_savedlr_FP104_QAT',
 'data_root': '/root/autodl-tmp/datasets/SIRST',
 'dataset_config': 'configuration.py',
 'parent_checkpoint': None,
 'parent_checkpoint_sha256': 'ceec7fc7460b6742fc3c3c384512654695918bb92482ab5d6756b7643f7040c3',
 'activation_ranges_path': None,
 'activation_ranges_sha256': '1402e81b54bcfa279485cca79c10ea167e3a64906d21dbb37c5d1af3569a84ea',
 'source_baseline': 'tianfang_acm_pytorch_quant_geometry_envelope_v3',
 'frozen_core_sha256': '85d75df1bf11fca2af71d94ae53286c5d746a0cd542281214ccc77ca12be12a3',
 'automatic_TEST': False,
 'authors_exact_configuration_verified': False,
 'initialization': 'Exact FP104 network and full parent Adagrad savedLR; fresh quantization Adam; '
                   'frozen TRAIN ranges',
 'packaging_status': 'Python-source-only example with external FP104 checkpoint, activation ranges '
                     'and dataset',
 'early_quality_stop': {'consecutive_epochs': 5,
                        'Pd_equal': 0,
                        'IoU_below': 0.01,
                        'action': 'Save completed epoch stable last/checkpoint/curve then clean '
                                  'exit; preserve failure, do not resume or extend'},
 'external_parent_checkpoint_required': True,
 'weights_bundled': False,
 'expected_initial_state': {'model': 'a9dde69aaf39147d550f80aa8806581e8b4e7fcb7deb590386c8a4eea0aed93b',
                            'optimizer': 'e343e9de704946e47a5fbe6874fe6d69852ce801c4732f3e519ad42200e4bdfe',
                            'RNG': '83768d63a238577b8b3b8af089a75df5b72bc3c53c52a7330ef6a084cc759eef'},
 'expected_first213_sha256': 'd0e45381e5f4c58641a40f3aadac0cd88851f6eb0fbb15aac4260ef5df8596db'}

DATASET_CONFIG = {'dataset': 'SIRST',
 'total_samples': 427,
 'paper_ratios': {'train': 0.5, 'val': 0.3, 'test': 0.2},
 'rounding': 'floor train and val; remainder to test',
 'counts': {'train': 213, 'val': 128, 'test': 86},
 'split_group': 'paper_ratio_seed0',
 'seed': 0,
 'seed_source': 'project choice; not specified in the paper',
 'authors_exact_sample_lists_verified': False,
 'model_selection_split': 'val',
 'final_evaluation_split': 'test'}

DATASET_PROVENANCE_SHA256 = '9a49f5c200281a02ee79c3fa950c69485d914cb549299ac662ee832d082a286d'

FLOPS_PROFILE = {'input_shape': [1, 3, 512, 512],
 'reference_bits': 32,
 'conv_counter': 1969816192,
 'bn_counter': 44957936,
 'pool_counter': 393264,
 'fp_reference_counter': 2015167392,
 'weight_quantizer_modules': 61,
 'activation_quantizer_modules': 52,
 'bitwidth_statistic': 'discrete_module_mean'}

REQUIRED_PACKAGES = ['torch==1.13.0',
 'torchvision==0.14.0',
 'numpy==1.23.5',
 'Pillow==9.4.0',
 'matplotlib==3.7.1',
 'tqdm==4.61.2',
 'PyYAML==6.0',
 'tensorboardX==2.6.2.2',
 'protobuf==3.20.3',
 'opencv-python-headless==4.8.1.78',
 'scipy==1.10.1',
 'scikit-image==0.21.0',
 'thop==0.1.1.post2209072238']

def get_training_config():
    from source_manifest import MANIFEST
    config = deepcopy(TRAINING)
    config['source_file_sha256'] = {name: entry['sha256'] for name, entry in MANIFEST.items()
                                    if name.endswith('.py')}
    return config
