"""Validation-only reports; merge compatible sessions without duplicating seeds."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
import zipfile

from ..checkpoint import read_checkpoint_metadata
from ..data import write_json
from .cycle2_report import collect_rows as cycle2_rows
from .cycle3_protocol import ARMS, COHORTS, SEEDS, PROTOCOL_ID


def collect_rows(campaign):
    rows, curves = [], []
    for job in campaign.manifest['runs']:
        if job['status'] != 'complete': continue
        a,b = cycle2_rows(SimpleNamespace(manifest={'runs':[job]}))
        row = a[0]
        row.update(arm=job['arm'], max_steps=job['max_steps'],
                   relative_flop_budget_shortfall=job['relative_flop_budget_shortfall'],
                   checkpoint_model_sha256=job['checkpoint_model_sha256'],
                   environment=read_checkpoint_metadata(job['checkpoint'])['environment'])
        rows.append(row)
        for value in b:
            value.update(arm=job['arm'], training_targets=value['step']*job['batch_size']*job['grad_accum_steps']*
                         job['config_resolved']['data']['seq_len'],
                         training_flops_estimated=value['step']*job['step_flops_estimated'])
        curves.extend(b)
    return rows,curves


def effect(values):
    n = len(values)
    mean = statistics.mean(values) if values else None
    sd = statistics.stdev(values) if n>=2 else None
    critical = {2:12.706204736, 3:4.302652730, 4:3.182446305, 5:2.776445105}.get(n)
    margin = critical*sd/n**.5 if critical is not None else None
    interval = [mean-margin,mean+margin] if margin is not None else None
    return {'n':n,'mean':mean,'sd':sd,'ci95_paired_t':interval,
            'seeds_favoring_flrm':sum(v<0 for v in values)}


def runtime_contract(environment):
    # Aggregation is not checkpoint resumption. Keep runtime/hardware type fixed,
    # while allowing physical GPU UUIDs, slot masks and host CPU identities to vary.
    return {key:environment[key] for key in ('python','packages','cuda_version','cudnn_version','device','cublas_workspace_config')}


def analyze(rows, curves, *, identity, smoke=False, tests_skipped=False):
    lookup = {(r['arm'],r['seed']):r for r in rows}
    if len(lookup)!=len(rows): raise ValueError('duplicate arm/seed; replicas are not independent')
    if any(r['seed'] not in SEEDS or r['arm'] not in ARMS for r in rows):
        raise ValueError('unknown arm or seed')
    if len({json.dumps(runtime_contract(r['environment']),sort_keys=True) for r in rows})>1:
        raise ValueError('incompatible runtime versions or hardware types')
    complete = [s for s in SEEDS if all((a,s) in lookup for a in ARMS)]
    effects = {}
    for cohort,arms in COHORTS.items():
        for arm in arms:
            if arm=='anchor_flrm': continue
            values = [lookup['anchor_flrm',s]['validation_loss']-lookup[arm,s]['validation_loss'] for s in complete]
            effects[f'{cohort}:anchor_flrm-minus-{arm}'] = effect(values)
    primary = effects['matched_estimated_flops:anchor_flrm-minus-flops_transformer']
    eligible = len(complete)==5 and not smoke and not tests_skipped
    decision = 'inconclusive'
    if eligible:
        lower,upper = primary['ci95_paired_t']
        if primary['mean'] <= -.005 and upper<0 and primary['seeds_favoring_flrm']>=4:
            decision='quality_advantage_at_matched_estimated_flops'
        elif lower>=-.002 and upper<=.002:
            decision='equivalent_within_predeclared_margin'
        elif primary['mean']>=.005 and lower>0:
            decision='transformer_quality_advantage_at_matched_estimated_flops'
    return {'protocol_id':PROTOCOL_ID, 'identity':identity, 'rows':rows, 'curves':curves,
            'complete_seeds':complete,'pending_seeds':[s for s in SEEDS if s not in complete],
            'effects':effects,'primary':primary,'confirmation_eligible':eligible,
            'decision':decision,'engineering_smoke':smoke,'tests_skipped':tests_skipped,
            'test_evaluated':False,'anchor_reused_between_cohorts':True,
            'limitations':'Validation endpoint only; five complete seeds required. Primary paired t interval assumes independent seed differences. Secondary contrasts exploratory. FLOPs are estimates excluding elementwise/pooling/mixing; neither wall time nor parameters are equalized. Time to a predeclared quality target was not measured. Reserved test awaits validation selection.'}


def save_report(destination, result):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    for name,values in (('runs.csv',result['rows']),('curves.csv',result['curves'])):
        if values:
            with (destination/name).open('x',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write_json(destination/'report.json',result)
    lines=['# Ciclo 3 — confirmação independente','',
           f"Decisão: {result['decision']}. Seeds completas: {result['complete_seeds']}; pendentes: {result['pending_seeds']}.",
           '', 'A âncora FLRM participa dos dois ensaios; é um único treino por seed.', '',
           '| Ensaio/braço | Seed | Passos | NLL | Treino, s | FLOPs estimados |',
           '|---|---:|---:|---:|---:|---:|']
    for r in result['rows']:
        lines.append(f"| {r['arm']} | {r['seed']} | {r['max_steps']} | {r['validation_loss']:.6f} | {r['train_wall_seconds']:.1f} | {r['training_flops_estimated']:.0f} |")
    lines += ['',f"Contraste primário FLRM − Transformer: {json.dumps(result['primary'])}",'',result['limitations'],
              '', 'Na próxima sessão, use somente as seeds pendentes e outro run_number. Combine relatórios compatíveis com `python -m llm2.research.cycle3_report --inputs ... --output NOVA_PASTA`.']
    (destination/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if result['curves']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for axis,key,label in zip(axes,['training_targets','training_flops_estimated'],['Alvos de treino','FLOPs estimados de treino']):
            for arm,seed in sorted({(r['arm'],r['seed']) for r in result['curves']}):
                values=sorted([r for r in result['curves'] if (r['arm'],r['seed'])==(arm,seed)],key=lambda r:r['step'])
                axis.plot([r[key] for r in values],[r['validation_loss'] for r in values],label=f'{arm}/{seed}',alpha=.7)
            axis.set_xlabel(label);axis.set_ylabel('NLL de validação');axis.grid(alpha=.2)
        axes[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(destination/'quality_by_cost.png',dpi=150);plt.close(fig)


def write_report(campaign):
    rows,curves=collect_rows(campaign)
    identity={'protocol_id':PROTOCOL_ID,'dataset_sha256':campaign.manifest.get('dataset_sha256'),
              'source_sha256':campaign.manifest['provenance']['source_sha256'],
              'versioned_configs_sha256':campaign.manifest['provenance']['versioned_configs_sha256']}
    result=analyze(rows,curves,identity=identity,smoke=campaign.manifest['engineering_smoke'],
                   tests_skipped=campaign.manifest['tests_skipped'])
    save_report(campaign.output/'report',result)
    return result


def merge(inputs, output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    reports=[]
    for file in map(Path,inputs):
        if file.suffix=='.zip':
            with zipfile.ZipFile(file) as archive:
                reports.append(json.loads(archive.read('report/report.json')))
        else:
            reports.append(json.loads(file.read_text(encoding='utf-8')))
    if not reports: raise ValueError('no reports')
    if any(r['identity']!=reports[0]['identity'] or r['protocol_id']!=PROTOCOL_ID or r['test_evaluated'] for r in reports):
        raise ValueError('incompatible protocol, dataset or source')
    rows=[row for r in reports for row in r['rows']]
    result=analyze(rows,[v for r in reports for v in r['curves']],identity=reports[0]['identity'],
                   smoke=any(r['engineering_smoke'] for r in reports),tests_skipped=any(r['tests_skipped'] for r in reports))
    save_report(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();merge(a.inputs,a.output)
