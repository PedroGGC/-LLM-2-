"""Contextual language worker and transparent order/streaming diagnostic."""
import argparse
import torch
from . import context_protocol
from .stories_worker import run


@torch.no_grad()
def context_diagnostic(trainer,*,smoke=False):
    model=trainer.model;was=model.training;model.eval()
    a,b,tail=b'xxcatxx',b'xxdogxx',b'xxxxxxxx'
    ids=torch.tensor([[256,*(a+b+tail)],[256,*(b+a+tail)]],device=trainer.device)
    try:
        expected=model(ids).logits
        value={'scope':'frozen order-collision probe; sensitivity, not semantic accuracy',
            'prefixes':[(a+b).decode(),(b+a).decode()],'common_suffix':tail.decode(),
            'final_logit_difference':float((expected[0,-1]-expected[1,-1]).abs().max())}
        if trainer.config.research.architecture.startswith('context_'):
            state=None;pieces=[]
            for segment in ids.split(3,1):
                output=model(segment,state=state);state=output.state;pieces.append(output.logits)
            actual=torch.cat(pieces,1);torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-3)
            value['streaming_logit_max_error']=float((actual-expected).abs().max())
        return value
    finally:model.train(was)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--variant',required=True);p.add_argument('--campaign-root',required=True)
    p.add_argument('--mode',choices=['train'],default='train');p.add_argument('--engineering-smoke',action='store_true')
    p.add_argument('--work-deadline',type=float);p.add_argument('--profile-permit');a=p.parse_args()
    run(a.config,a.variant,a.campaign_root,smoke=a.engineering_smoke,deadline=a.work_deadline,permit=a.profile_permit,
        recipe=context_protocol,diagnostic=context_diagnostic)

if __name__=='__main__':main()
