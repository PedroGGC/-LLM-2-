"""Frozen contextual memory/depth/hierarchy screen after campaign 0032."""
import hashlib
import json
from pathlib import Path
from ..config import ModelConfig,ResearchConfig
from . import stories_protocol as previous
from .stories_data import PINNED_FINGERPRINT

VARIANTS=('transformer_matched','context_single','transformer_medium','context_shared2',
          'context_untied2','hierarchy_shared','hierarchy_untied','transformer')
SEEDS=(341,342)
RATES=(.0003,.001)
CALIBRATION_SEED=941
PROMPTS=previous.PROMPTS
CYCLE='context_language_v1'
WORKER_MODULE='llm2.research.context_worker'
REFERENCES=('transformer_matched','transformer_medium','transformer')
CPU_TESTS=('tests/test_context_memory.py','tests/test_context_training.py','tests/test_stories_data.py')
CUDA_TESTS=('tests/test_context_memory.py',)
PROTOCOL={**previous.PROTOCOL,'name':'contextual-retention-language-v1','variants':VARIANTS,'seeds':SEEDS,
 'calibration_seed':CALIBRATION_SEED,'dataset':PINNED_FINGERPRINT,
 'architecture':'content-gated scalar decay per head; local causal representations; residual contextual depth',
 'hierarchy':'fine per byte, contextual mean per4 completed positions, same or independent core',
 'optimization':'local32 attention matrices plus affine parallel block scan; no global SDPA; O(N log N) block scan work',
 'sharing':'same dimensions/operations; copied initial independent cores; separate runtime states',
 'cost_targets':[2,4,8,20,100,1000,1000000],
 'cost_acceptance':'at least 2x isolated training update speed and NLL within+0.002 in BOTH seeds versus named reference; exploratory, not confirmation'}
PROTOCOL_ID=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def configuration(variant,data,output,seed,*,lr=.0003,smoke=False,device='cuda',calibration=False):
    if variant not in VARIANTS or seed not in SEEDS or lr not in RATES:raise ValueError('unknown contextual arm/seed/rate')
    old=variant if variant in ('transformer','transformer_matched') else 'transformer_matched'
    cfg=previous.configuration(old,data,output,331,lr=lr,smoke=smoke,device=device,calibration=calibration)
    cfg.train.seed=CALIBRATION_SEED if calibration else seed
    if variant=='transformer_medium' and not smoke:
        cfg.model=ModelConfig(vocab_size=257,d_model=80,n_layers=2,n_heads=4,hidden_dim=160,max_seq_len=8192,dropout=.05)
    elif not variant.startswith('transformer'):
        cfg.model=ModelConfig(vocab_size=257,d_model=16 if smoke else 80,n_layers=1 if variant=='context_single' else 2,
            n_heads=2 if smoke else 4,hidden_dim=24 if smoke else 144,max_seq_len=64 if smoke else 8192,dropout=.05)
        cfg.research=ResearchConfig(architecture='context_hierarchy' if variant.startswith('hierarchy') else 'context_memory',
            share_weights='untied' not in variant,scale_conditioning=False,memory_dim=4 if smoke else 16,state_chunk_size=32)
    return cfg


def jobs(output,data,gpus,*,seeds=SEEDS,smoke=False,device='cuda'):
    return previous.jobs(output,data,gpus,seeds=seeds,smoke=smoke,device=device,variants=VARIANTS,config_factory=configuration)
