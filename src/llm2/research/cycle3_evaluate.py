"""Evaluate a run's checkpoints in one process, using the existing CLI unchanged."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from ..cli import main as cli_main, _inference
from ..data import validation_batches, write_json
from ..evaluation import evaluate
from ..objectives import CausalLanguageModeling


def bundle(config, directory, *, device='cuda', timing_only=False):
    directory=Path(directory)
    if timing_only:
        if directory.exists(): raise FileExistsError(directory)
        directory.mkdir(parents=True)
        cfg,model,data,target,_=_inference(argparse.Namespace(config=config,checkpoint=None,data=None,
                                                             device=device,precision='fp32'))
        started=time.monotonic()
        if target.type=='cuda':torch.cuda.synchronize(target)
        metrics=evaluate(model,CausalLanguageModeling(),validation_batches(data.arrays['val'],cfg.data.seq_len,
                         cfg.data.eval_batch_size,None,target),device=target,precision='fp32',deterministic=cfg.train.deterministic)
        if target.type=='cuda':torch.cuda.synchronize(target)
        result={'validation_seconds':time.monotonic()-started,'validation_tokens':metrics['validation_tokens'],
                'split':'val','loss_not_recorded':True,'purpose':'duration pilot only, no selection by quality'}
        write_json(directory/'timing.json',result)
        return result
    from ..config import load_config
    cfg=load_config(config)
    run=Path(cfg.train.output_dir)
    if directory.resolve() != run.resolve():
        raise ValueError('evaluation directory must match the configured run')
    checkpoints=sorted((run/'checkpoints').glob('step-*'),key=lambda p:int(p.name.split('-')[-1]))
    expected=[cfg.train.max_steps*i//4 for i in range(1,5)]
    if [int(p.name.split('-')[-1]) for p in checkpoints] != expected:
        raise ValueError('missing or unexpected milestone checkpoints')
    records=[]
    for checkpoint in checkpoints:
        destination=run/'evaluations'/checkpoint.name
        destination.mkdir(parents=True,exist_ok=False)
        common=['--checkpoint',str(checkpoint),'--device',device,'--precision','fp32']
        started=time.monotonic()
        cli_main(['evaluate',*common,'--split','val','--output',str(destination/'language.json'),
                  '--windows-output',str(destination/'windows.jsonl')])
        value=json.loads((destination/'language.json').read_text(encoding='utf-8'))
        records.append({'step':int(checkpoint.name.split('-')[-1]),'checkpoint':str(checkpoint),
                        'checkpoint_model_sha256':value['checkpoint_model_sha256'],
                        'validation_loss':value['validation_loss'],'validation_tokens':value['validation_tokens'],
                        'evaluation':str(destination/'language.json'),'wall_seconds':time.monotonic()-started})
        if checkpoint==checkpoints[-1]:
            cli_main(['benchmark',*common,'--prompt','The history of','--warmup-tokens','8','--max-new-tokens','64',
                      '--output',str(destination/'benchmark.json')])
            cli_main(['diagnose',*common,'--samples','128','--batch-size','4','--output',str(destination/'diagnostics.json')])
        gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    result={'checkpoints':records,'test_evaluated':False,'split':'val'}
    write_json(run/'evaluation_bundle.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--directory',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--timing-only',action='store_true')
    a=p.parse_args(argv)
    bundle(a.config,a.directory,device=a.device,timing_only=a.timing_only)


if __name__=='__main__':main()
