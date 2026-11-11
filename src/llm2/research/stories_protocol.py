"""Fixed first language screen; learning rates selected on train-derived calibration."""
import hashlib
import json
from pathlib import Path
from ..config import ExperimentConfig,ModelConfig,DataConfig,TrainConfig,OptimizerConfig,ResearchConfig
from .stories_data import PINNED_FINGERPRINT

VARIANTS=('transformer','transition','pyramid_shared','transformer_matched')
SEEDS=(331,332)
RATES=(.0003,.001)
CALIBRATION_SEED=931
CYCLE='stories_language_v1'
WORKER_MODULE='llm2.research.stories_worker'
REFERENCES=('transformer','transformer_matched')
CPU_TESTS=('tests/test_stories_models.py','tests/test_stories_data.py','tests/test_stories_training.py')
CUDA_TESTS=('tests/test_stories_models.py',)
PROMPTS=('Once upon a time, there was a little rabbit',
         'Lily found a small key near the old tree.',
         'Tom wanted to help his friend, but')
PROTOCOL={'name':'tinystories-language-v1','dataset':PINNED_FINGERPRINT,'variants':VARIANTS,'seeds':SEEDS,
 'steps':6000,'context':256,'batch':32,'precision':'fp32','dropout':.05,'weight_decay':.01,'warmup':100,
 'rates':RATES,'calibration_steps':400,'calibration_seed':931,'rate_selection':'full cal_val NLL; lower LR on exact tie; fresh main init',
 'evaluation_steps':[1500,3000,4500,6000],'validation':'all selected development bytes, per-document windows, no test',
 'prompts':PROMPTS,'generation_bytes':256,'temperature':.8,'top_k':40,'generation_seed':13931,
 'comparison':'equal sampled windows/targets; not matched compute; matched parameter Transformer also provided',
 'optimization':'logical group4, compute block32; historical reference preserved',
 'target_quality_nll':.005,'equivalence_nll':.002,'efficiency_fraction':.10}
PROTOCOL_ID=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def configuration(variant,data,output,seed,*,lr=.0003,smoke=False,device='cuda',calibration=False):
    if variant not in VARIANTS or seed not in SEEDS or lr not in RATES:raise ValueError('unknown language arm/seed/rate')
    d,h,k=(16,24,8) if smoke else (96,128,64)
    model=ModelConfig(vocab_size=257,d_model=d,n_layers=1,n_heads=2 if smoke else 4,hidden_dim=h,
        max_seq_len=64 if smoke else 8192,dropout=.05)
    research=ResearchConfig(architecture='transition_memory',scale_conditioning=False,memory_dim=k,
        associative_levels=2 if variant=='pyramid_shared' else 1,associative_execution='blocked',associative_block_size=32)
    if variant.startswith('transformer'):
        research=ResearchConfig(architecture='transformer',scale_conditioning=False)
        if not smoke:
            model=ModelConfig(vocab_size=257,d_model=384 if variant=='transformer' else 64,n_layers=2,
                n_heads=6 if variant=='transformer' else 4,hidden_dim=384 if variant=='transformer' else 96,
                max_seq_len=8192,dropout=.05)
    steps=(2 if calibration else 4) if smoke else (400 if calibration else 6000)
    return ExperimentConfig(model=model,data=DataConfig(path=str(Path(data).resolve()),seq_len=32 if smoke else 256,
        batch_size=2 if smoke else 32,eval_batch_size=2 if smoke else 32),
        optimizer=OptimizerConfig(lr=lr,weight_decay=.01,warmup_steps=1 if smoke else 50 if calibration else 100,schedule_steps=steps),
        train=TrainConfig(seed=931 if calibration else seed,device=device,precision='fp32',deterministic=device=='cpu',compile=not smoke,
            max_steps=steps,grad_accum_steps=2 if smoke else 1,eval_interval=steps if calibration else max(1,steps//4),
            checkpoint_interval=steps if calibration else max(1,steps//2),log_interval=1 if smoke else 300,
            output_dir=str(Path(output).resolve()),cpu_threads=2,accumulation_weighting='tokens',console_progress=not calibration),research=research)


def jobs(output,data,gpus,*,seeds=SEEDS,smoke=False,device='cuda',variants=VARIANTS,config_factory=configuration):
    rows=[]
    for si,seed in enumerate(seeds):
        for vi,variant in enumerate(variants):
            name=f'{variant}-seed-{seed}';cfg=config_factory(variant,data,Path(output)/'runs'/name,seed,smoke=smoke,device=device)
            rows.append({'id':name,'variant':variant,'model':variant,'seed':seed,'directory':cfg.train.output_dir,
                'config_resolved':cfg.to_dict(),'gpu_slot':str(gpus[(vi+si)%2]),'physical_gpu':None if device=='cpu' else int(gpus[(vi+si)%2]),
                'status':'pending','max_steps':cfg.train.max_steps,'batch_size':cfg.data.batch_size,'grad_accum_steps':cfg.train.grad_accum_steps,
                'checkpoint_interval':cfg.train.checkpoint_interval,'progress_interval':cfg.train.log_interval})
    return rows
