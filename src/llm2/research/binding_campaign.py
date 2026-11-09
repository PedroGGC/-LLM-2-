"""Two-hour binding campaign with persistent workers and isolated final profiles."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import html
import json
import os
from pathlib import Path
import shutil
import time
import traceback
STARTED=time.monotonic()
import torch
from ..checkpoint import read_checkpoint_metadata,resume_contract
from ..config import ExperimentConfig
from ..data import write_json,sha256_file
from ..runtime import environment,save_source_snapshot
from .binding_protocol import VARIANTS,SEEDS,PROTOCOL,PROTOCOL_ID,configuration,jobs
from .binding_data import TASK_ID
from .campaign_storage import verify_pair
from .cycle3_short import heartbeat
from .cycle4 import archive_files,checkpoint_files
from .session import SessionBudget,SessionBudgetExceeded,SessionCampaign
from .state_study import materialize,storage_budget,discard_compiler_cache

ROOT=Path(__file__).resolve().parents[3]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def native_history(campaign,args):
    result={};runtime=environment()
    for path in sorted(campaign.output.parent.glob('llm2-campaign-*/campaign.json')):
        if path.parent==campaign.output or path.is_symlink():continue
        try:value=read(path)
        except (OSError,ValueError):continue
        if value.get('cycle')!='transition_binding_v1' or value.get('protocol_id')!=PROTOCOL_ID:continue
        if value.get('engineering_smoke')!=args.engineering_smoke:continue
        if value['provenance']['source_sha256']!=campaign.manifest['provenance']['source_sha256']:
            raise ValueError('binding history source mismatch')
        if any(value['environment'].get(k)!=runtime.get(k) for k in ('python','packages','cuda_version','cudnn_version')):
            raise ValueError('binding history runtime mismatch')
        for job in value['runs']+value.get('historical_runs',[]):
            if job['status']!='complete':continue
            expected=configuration(job['variant'],'unused',job['seed'],smoke=args.engineering_smoke,device=args.device)
            if resume_contract(expected)!=resume_contract(ExperimentConfig.from_dict(job['config_resolved'])):
                raise ValueError('binding history configuration mismatch')
            origin=Path(job.get('historical_campaign',path.parent));cp=Path(job['checkpoint'])
            if (origin.is_symlink() or cp.is_symlink() or not origin.resolve().is_relative_to(campaign.output.parent.resolve())
                    or not cp.resolve().is_relative_to(origin.resolve())):raise ValueError('binding history outside working')
            meta=read_checkpoint_metadata(cp)
            if (job['step']!=expected.train.max_steps or meta['config']!=job['config_resolved'] or
                meta['dataset_sha256']!=TASK_ID or meta['files']['model.safetensors']!=job['checkpoint_model_sha256'] or
                meta['provenance']['source_sha256']!=campaign.manifest['provenance']['source_sha256']):
                raise ValueError('binding history checkpoint mismatch')
            if job['id'] in result and result[job['id']]['checkpoint_model_sha256']!=job['checkpoint_model_sha256']:
                raise ValueError('conflicting same-seed binding history')
            result[job['id']]={**job,'historical_campaign':str(origin),'history_manifest_sha256':sha256_file(origin/'campaign.json')}
    return list(result.values())


def run_pair(campaign,args,pair):
    """No training on either lane while the final benchmark permit is issued."""
    def run(job):
        job['status']='running';campaign.save()
        cfg=ExperimentConfig.from_dict(job['config_resolved'])
        path=materialize(cfg,campaign.output/'configs'/(job['id']+'.yaml'))
        cmd=['-m','llm2.research.binding_worker','--config',path,'--mode','train',
             '--work-deadline',str(time.monotonic()+campaign.budget.work_remaining),
             '--profile-permit',str(Path(job['directory'])/'profile-permit.json')]
        if args.engineering_smoke:cmd.append('--engineering-smoke')
        start=time.monotonic()
        campaign.command(cmd,job['id']+'.log',gpu=None if args.engineering_smoke else job['gpu_slot'],job=job)
        value=read(Path(job['directory'])/'worker.json')
        if value['protocol_id']!=PROTOCOL_ID or value['task_sha256']!=TASK_ID or value['source_sha256']!=campaign.manifest['provenance']['source_sha256']:
            raise ValueError('binding worker provenance mismatch')
        if value['status']=='complete' and value['step']!=job['max_steps']:raise ValueError('binding worker horizon mismatch')
        with campaign.lock:
            job.update(value,command_wall_seconds=time.monotonic()-start);campaign.save()
    futures={};reported={};active=None
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={j['id']:pool.submit(run,j) for j in pair}
        try:
            while not all(f.done() for f in futures.values()):
                for f in futures.values():
                    if f.done() and f.exception() is not None:f.result()
                for j in pair:
                    root=Path(j['directory']);forecast=root/'forecast.json'
                    if forecast.exists() and not reported.get((j['id'],'forecast')):
                        try:v=read(forecast)
                        except (ValueError,OSError):continue
                        print(f"PILOTO {j['id']}: {1000*v['pilot']['median_step_seconds']:.2f}ms/passo; previsão{v['forecast_remaining_seconds']/60:.1f}min",flush=True)
                        reported[j['id'],'forecast']=True
                    point=root/'binding-progress.json'
                    if point.exists():
                        try:v=read(point)
                        except (ValueError,OSError):continue
                        if reported.get((j['id'],'step'))!=v['step']:
                            print(f"ACERTO {j['id']} passo{v['step']}: {100*v['accuracy']:.2f}% | par contrafactual{100*v['paired_accuracy']:.2f}%",flush=True)
                            reported[j['id'],'step']=v['step']
                if active is not None and futures[active].done():active=None
                ready=all(futures[j['id']].done() or (Path(j['directory'])/'profile-ready.json').exists() for j in pair)
                if ready and active is None:
                    next_job=next((j for j in pair if not futures[j['id']].done()),None)
                    if next_job:
                        active=next_job['id']
                        write_json(Path(next_job['directory'])/'profile-permit.json',{'scope':'both lanes finished training/evaluation; one profile at a time'})
                        print(f"PERFIL ISOLADO: {active}",flush=True)
                time.sleep(.25)
            for f in futures.values():f.result()
        except BaseException:campaign.abort();raise


def report(campaign):
    evidence=campaign.manifest.get('historical_runs',[])+campaign.manifest['runs'];rows=[];curves=[]
    for job in evidence:
        for e in job.get('evaluations',[]):
            curves.append({'variant':job['variant'],'seed':job['seed'],'step':e['step'],'accuracy':e['accuracy'],
                           'paired_accuracy':e['paired_accuracy'],'query_nll':e['query_nll'],'training_queries':e['training_queries'],
                           'training_wall_seconds':e.get('training_wall_seconds'),'experiment_wall_seconds':e.get('experiment_wall_seconds')})
        for e in job.get('final',[]):
            p=next((p for p in job.get('profiles',[]) if p['length']==e['length']),{})
            energy=p.get('energy',{})
            rows.append({'variant':job['variant'],'seed':job['seed'],'length':e['length'],'accuracy_percent':100*e['accuracy'] if e.get('accuracy') is not None else None,
                'counterfactual_percent':100*e['counterfactual_accuracy'] if e.get('accuracy') is not None else None,
                'paired_percent':100*e['paired_accuracy'] if e.get('accuracy') is not None else None,
                'queries':e.get('queries'),'query_nll':e.get('query_nll'),'error':e.get('error'),
                'parameters':job.get('parameters'),'physical_gpu':job['gpu_slot'],
                'median_forward_ms':1000*p['median_forward_seconds'] if p else None,
                'state_bytes':p.get('returned_state_bytes'),'peak_forward_vram_bytes':p.get('peak_forward_vram_bytes'),
                'gpu_joules_per_query':energy.get('gpu_joules_per_query'),
                'train_step_ms':1000*job['training_profile']['median_step_seconds'] if job.get('training_profile') else None})
    decisions=[]
    for row in rows:
        if row['variant']=='transformer':continue
        base=next((r for r in rows if r['variant']=='transformer' and r['seed']==row['seed'] and r['length']==row['length']),None)
        qualified=row['accuracy_percent'] is not None and row['accuracy_percent']>=95 and row['paired_percent']>=90
        reference_qualified=base is not None and base['accuracy_percent'] is not None and base['accuracy_percent']>=95 and base['paired_percent']>=90
        comparable=qualified and reference_qualified and row['accuracy_percent']>=base['accuracy_percent']-1
        ratio=base['parameters']/row['parameters'] if base else None
        speed=base['median_forward_ms']/row['median_forward_ms'] if base and base['median_forward_ms'] and row['median_forward_ms'] else None
        decisions.append({'variant':row['variant'],'seed':row['seed'],'length':row['length'],'qualified_binding':qualified,
            'reference_qualified':reference_qualified,
            'retrieval_advantage_in_this_recipe':bool(qualified and base and base['accuracy_percent'] is not None and row['accuracy_percent']>=base['accuracy_percent']+10),
            'parameter_factor':ratio,'forward_speed_factor':speed,'compactness_candidate':bool(comparable and ratio>=2),
            'forward_efficiency_candidate':bool(comparable and speed is not None and speed>=2),
            'scope':'exploratory task-specific candidate; not general LLM superiority or confirmation'})
    folder=campaign.output/'report';folder.mkdir(exist_ok=True)
    value={'status':campaign.manifest['status'],'protocol_id':PROTOCOL_ID,'task_sha256':TASK_ID,'rows':rows,'decisions':decisions,
        'capacity':[{"variant":j['variant'],"seed":j['seed'],**j['capacity']} for j in evidence if 'capacity' in j],
        'alignment':[{"variant":j['variant'],"seed":j['seed'],**j['alignment_probe']} for j in evidence if 'alignment_probe' in j],
        'limitations':['lag-one inductive bias and adjacent records','synthetic unique-map retrieval, not language',
        'fixed common recipe, not matched compute or per-family tuning','different parameters','two seeds, exploratory',
        'eager full forward at each measured length, not generation/KV','GPU board energy excludes host/setup/compile',
        'FLOPs of transition models are null','complete native history only; incomplete jobs preserved but not auto-resumed']}
    write_json(folder/'report.json',value)
    for label,data in (('scores',rows),('curves',curves)):
        if data:
            with (folder/(label+'.csv')).open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    keys=('variant','seed','length','accuracy_percent','paired_percent','parameters','median_forward_ms')
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in keys)+'</tr>' for r in rows)
    page='<meta charset="utf-8"><title>Associação aprendida</title><style>body{font:16px sans-serif;margin:30px}td,th{padding:9px;border-bottom:1px solid #ccc}</style>'
    page+='<h1>Memória de transições e pirâmide compartilhada</h1><p>Recuperação sintética, com perguntas contrafactuais. Compactação é diferente de eficiência. Tempos são forwards completos, não geração/KV.</p><table><tr>'
    page+=''.join('<th>'+k+'</th>' for k in keys)+'</tr>'+table+'</table>'
    (folder/'report.html').write_text(page,encoding='utf-8')
    return value


def publish(campaign):
    campaign.save();report(campaign)
    files=[p for p in campaign.output.rglob('*') if p.is_file() and 'compiler-cache' not in p.parts and
        p.suffix in ('.json','.jsonl','.yaml','.log','.txt','.csv','.html','.zip') and
        (p.name!='source.zip' or p.parent==campaign.output)]
    results=archive_files(campaign.output.parent/(campaign.output.name+'-results.zip'),files,base=campaign.output)
    weights=archive_files(campaign.output.parent/(campaign.output.name+'-checkpoints.zip'),checkpoint_files(campaign,campaign.manifest['runs']),base=campaign.output)
    verify_pair(results,weights);return results,weights


def execute(args):
    budget=SessionBudget(args.session_seconds,300,elapsed=args.bootstrap_elapsed_seconds+time.monotonic()-args.started)
    campaign=SessionCampaign(ROOT,args.output,device=args.device,gpus=args.gpus,budget=budget)
    campaign.manifest.update(cycle='transition_binding_v1',protocol=PROTOCOL,protocol_id=PROTOCOL_ID,dataset_sha256=TASK_ID,
        task_sha256=TASK_ID,engineering_smoke=args.engineering_smoke,environment=environment(),historical_runs=[],
        limitations='task-specific association, not language; no automatic general superiority claim')
    old={k:os.environ.get(k) for k in ('LLM2_CYCLE4','LLM2_COMPILER_CACHE_ROOT')}
    os.environ.update(LLM2_CYCLE4='1',LLM2_COMPILER_CACHE_ROOT=str(campaign.output/'compiler-cache'))
    error=None
    try:
        with storage_budget(campaign,smoke=args.engineering_smoke),heartbeat(campaign,'associação e pirâmide'):
            if args.bootstrap_metadata:
                meta=read(args.bootstrap_metadata)
                for name,digest in meta['source_files_sha256'].items():
                    if sha256_file(ROOT/name)!=digest:raise ValueError('bootstrap source changed: '+name)
                if meta['source_git_revision']!=campaign.manifest['provenance']['git']['commit']:raise ValueError('bootstrap revision mismatch')
                write_json(campaign.output/'bootstrap.json',meta)
            save_source_snapshot(campaign.output/'source.zip',campaign.manifest['provenance'])
            if not args.engineering_smoke:
                campaign.command(['-m','pytest','-q','tests/test_binding_models.py','-m','not cuda'],'tests-binding-cpu.log')
                campaign.command(['-m','pytest','-q','tests/test_binding_models.py','-m','cuda'],'tests-binding-cuda.log',gpu=args.gpus[0])
            history=native_history(campaign,args);campaign.manifest['historical_runs']=history
            done={j['id'] for j in history}
            rows=[j for j in jobs(campaign.output,args.gpus,seeds=args.seeds,smoke=args.engineering_smoke,device=args.device) if j['id'] not in done]
            campaign.manifest['runs']=rows;campaign.save()
            with torch.device('meta'):
                from .factory import build_backbone
                for j in rows:j['parameters']=sum(p.numel() for p in build_backbone(ExperimentConfig.from_dict(j['config_resolved'])).parameters())
            required=sum(j['parameters']*48 for j in rows)+(0 if args.engineering_smoke else 4*1024**3)
            if shutil.disk_usage(campaign.output).free<required:raise OSError('insufficient disk including 2GiB cache and 2GiB free reserve')
            campaign.manifest['storage_estimate']={'required_free_bytes':required}
            for seed in args.seeds:
                pending=[j for j in rows if j['seed']==seed]
                # Pair distinct GPU assignments; do not create repeated pilot workers.
                while pending:
                    budget.require(0 if args.engineering_smoke else 180)
                    first=pending.pop(0);second=next((j for j in pending if j['gpu_slot']!=first['gpu_slot']),None)
                    pair=[first]
                    if second:pending.remove(second);pair.append(second)
                    run_pair(campaign,args,pair)
            expected={f'{v}-seed-{s}' for s in args.seeds for v in VARIANTS}
            completed={j['id'] for j in history+rows if j['status']=='complete'}
            campaign.manifest['status']='complete' if expected<=completed else 'partial_planned'
    except SessionBudgetExceeded as exc:campaign.manifest.update(status='partial_budget',error=str(exc))
    except Exception as exc:error=exc;campaign.manifest.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    finally:
        campaign.abort()
        for j in campaign.manifest['runs']:
            path=Path(j['directory'])/'worker.json'
            if path.exists():
                try:
                    value=read(path)
                    if value.get('source_sha256')==campaign.manifest['provenance']['source_sha256']:j.update(value)
                except (ValueError,OSError):pass
        try:discard_compiler_cache(campaign)
        except Exception as exc:campaign.manifest['cache_cleanup_error']=str(exc)
        campaign.manifest['pending_ids']=[f'{v}-seed-{s}' for s in args.seeds for v in VARIANTS if f'{v}-seed-{s}' not in
            {j['id'] for j in campaign.manifest.get('historical_runs',[])+campaign.manifest['runs'] if j['status']=='complete'}]
        campaign.manifest['session_budget']=budget.snapshot()
        try:results,weights=publish(campaign)
        finally:
            for k,v in old.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
        print(f"Status:{campaign.manifest['status']} | {results.name} | {weights.name}",flush=True)
    if error:raise RuntimeError('binding campaign failed; evidence exported') from error
    return campaign.manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--gpus',nargs=2,default=['0','1'])
    p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS));p.add_argument('--engineering-smoke',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--session-seconds',type=float,default=7200)
    p.add_argument('--bootstrap-elapsed-seconds',type=float,default=0);p.add_argument('--bootstrap-metadata')
    args=p.parse_args(argv);args.started=STARTED
    if not 300<args.session_seconds<=7200 or len(set(args.gpus))!=2 or (args.device=='cpu')!=args.engineering_smoke:
        raise ValueError('two-hour bound/two slots required; CPU only for engineering smoke')
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or not set(args.seeds)<=set(SEEDS):raise ValueError('invalid seeds')
    if not args.engineering_smoke and (not args.bootstrap_metadata or not torch.cuda.is_available() or
        any(not s.isdecimal() or int(s)>=torch.cuda.device_count() or 'T4' not in torch.cuda.get_device_name(int(s)) for s in args.gpus)):
        raise ValueError('science requires pinned bootstrap and two T4 GPUs')
    return execute(args)


if __name__=='__main__':main()
