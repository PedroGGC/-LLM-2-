"""Predeclared independent-corpus confirmation; no new model mechanisms."""
import hashlib
import json
from pathlib import Path

from ..config import ExperimentConfig, load_config
from ..data import PreparedData
from .cycle2_protocol import configuration as cycle2_config, configuration_stats, paired_fingerprints
from .wikitext import REVISION, FILES, TRAIN_EXCLUSIONS

SEEDS = (101, 102, 103, 104, 105)
MODELS = ('baseline_small', 'lrm_no_jepa_k2', 'flrm_exchange_no_film', 'transformer_deep')
ARMS = ('targets_transformer', 'targets_lrm', 'anchor_flrm',
        'flops_transformer', 'flops_lrm', 'flops_transformer_deep')
ARM_MODEL = dict(zip(ARMS, (*MODELS[:3], MODELS[0], MODELS[1], MODELS[3])))
COHORTS = {'equal_targets': ARMS[:3], 'matched_estimated_flops': (ARMS[3], ARMS[4], ARMS[2], ARMS[5])}
TRAIN_BYTES = 32*1024**2
# Filled from the pinned real-corpus preparation before release, never from validation quality.
DATA_HASHES = {
    'test.bin': 'ba26e93bca72e59dbd2f12a6b1758fdafb97ce3b652310cb44de218ff91536cf',
    'tokenizer.json': 'f88f84a387d00bb8027db376d50fff07c86c4ee137f1678351521fefe0f8d252',
    'train.bin': '0d4d9109a5acd6820a3a050a20da2f2073a3d5f0da2b25138c9c47a9bd28906c',
    'val.bin': 'b2ce29824a4e56d51d396231f467c22e0ac3cb8916f5a01c69833207f4340b97',
}
PROTOCOL = {'cycle': 3, 'dataset_revision': REVISION, 'train_bytes': TRAIN_BYTES,
            'seeds': list(SEEDS), 'reference_steps': 8000, 'context': 256, 'batch': 32,
            'precision': 'fp32', 'compiler': 'inductor', 'context_normalization': 'none',
            'film': False, 'exchange': True, 'iterations': 2,
            'cohorts': COHORTS, 'primary': 'matched_estimated_flops:anchor_flrm-minus-flops_transformer',
            'rounding': 'floor to multiple of four', 'test_evaluated': False,
            'dataset_files': DATA_HASHES, 'training_overlap_exclusions': sorted(TRAIN_EXCLUSIONS),
            'json_serialization': 'UTF-8, sorted keys, indent 2, LF and final newline',
            'quality_margin': .005, 'equivalence_margin': .002,
            'deep_transformer': {'n_layers': 2, 'hidden_dim': 1024}}
PROTOCOL_ID = hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
ROOT = Path(__file__).resolve().parents[3]


def configuration(model, data, output, seed, *, steps=None, device='cuda', smoke=False):
    steps = (4 if smoke else 8000) if steps is None else steps
    if model not in MODELS or seed not in SEEDS:
        raise ValueError('unknown Cycle 3 model/seed')
    if type(steps) is not int or steps < 4 or steps % 4:
        raise ValueError('fixed horizon must be a positive multiple of four, at least four')
    base = load_config(ROOT/'configs/experiments/cycle3/base.yaml')
    alias = 'baseline_small' if model == 'transformer_deep' else model
    cfg = cycle2_config(base,alias,data,output,seed,device=device,smoke=smoke)
    if model == 'transformer_deep':
        cfg.model.n_layers = 2
        cfg.model.hidden_dim = 64 if smoke else 1024
    cfg.train.max_steps = cfg.optimizer.schedule_steps = steps
    cfg.train.checkpoint_interval = steps//4
    cfg.train.eval_interval = steps//4
    cfg.train.log_interval = steps//4
    cfg.research.scale_conditioning = False
    return ExperimentConfig.from_dict(cfg.to_dict())


def ordered_jobs(seeds, gpus, data, output, *, device='cuda', smoke=False):
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(SEEDS):
        raise ValueError('select unique seeds from 101..105')
    data, output = Path(data), Path(output)
    reference_steps = 4 if smoke else 8000
    stats = {m:configuration_stats(configuration(m,data,output/'meta',SEEDS[0],steps=reference_steps,
                                               device=device,smoke=smoke)) for m in MODELS}
    budget = stats[MODELS[2]]['step_flops_estimated']*reference_steps
    slots = list(gpus) if device == 'cuda' else [None]
    jobs=[]
    for seed in seeds:
        block=SEEDS.index(seed)
        order=ARMS[block%len(ARMS):]+ARMS[:block%len(ARMS)]
        for arm in order:
            model=ARM_MODEL[arm]
            steps = (max(4,int(budget//stats[model]['step_flops_estimated'])//4*4)
                     if arm.startswith('flops_') else reference_steps)
            total=steps*stats[model]['step_flops_estimated']
            if arm.startswith('flops_') and total > budget:
                raise ValueError('minimum discrete horizon exceeds FLOP budget')
            identity=f'{arm}-seed-{seed}'
            cfg=configuration(model,data,output/'runs'/identity,seed,steps=steps,device=device,smoke=smoke)
            jobs.append({'id':identity,'arm':arm,'model':model,'seed':seed,
                         'cohorts':[name for name,arms in COHORTS.items() if arm in arms],
                         'shared_anchor':arm=='anchor_flrm', 'status':'pending',
                         'gpu_slot':slots[(ARMS.index(arm)+block)%len(slots)],
                         'max_steps':steps,'checkpoint_interval':steps//4,'progress_interval':steps//4,
                         'iterations':cfg.research.iterations,'batch_size':cfg.data.batch_size,
                         'grad_accum_steps':cfg.train.grad_accum_steps,
                         'planned_training_flops_estimate':total,'reference_flop_budget_estimate':budget,
                         'relative_flop_budget_shortfall':(budget-total)/budget if arm.startswith('flops_') else 0.,
                         'config_resolved':cfg.to_dict(), **stats[model]})
    return jobs


def verify_dataset(path, *, smoke=False):
    data=PreparedData(path)
    if data.manifest['schema_version'] != 2 or set(data.arrays) != {'train','val','test'}:
        raise ValueError('Cycle 3 requires explicit whole-document train/val/test splits')
    if not smoke:
        corpus=data.manifest.get('corpus',{})
        if not DATA_HASHES:
            raise ValueError('Cycle 3 protocol has no pinned dataset hashes')
        expected = {
            **{f'files.{name}': digest for name, digest in DATA_HASHES.items()},
            'corpus.revision': REVISION, 'corpus.shard_sha256': FILES,
            'corpus.requested_train_bytes': TRAIN_BYTES,
            'corpus.training_overlap_exclusions': sorted(TRAIN_EXCLUSIONS),
        }
        observed = {
            **{f'files.{name}': digest for name, digest in data.manifest['files'].items()},
            **{key:corpus.get(key.split('.',1)[1]) for key in expected if key.startswith('corpus.')},
        }
        differences = [{'field': key, 'expected': expected.get(key), 'observed': observed.get(key)}
                       for key in sorted(set(expected)|set(observed)) if expected.get(key)!=observed.get(key)]
        if differences:
            raise ValueError('dataset does not match the pinned Cycle 3 corpus/content: '
                             + json.dumps(differences, sort_keys=True))
    return data
