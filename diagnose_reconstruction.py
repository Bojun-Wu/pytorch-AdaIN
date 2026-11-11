"""Diagnostic: alpha=0 still traverses AdaIN's encoder/decoder, not identity."""
import argparse
from pathlib import Path
import torch
from experiment_guard import Engine, CAL_CONTENT, write_csv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='guard_experiment')
    p.add_argument('--size',type=int,default=192)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    args.alpha=0.
    torch.set_num_threads(2)
    torch.manual_seed(0)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.inference_mode():
        engine=Engine(args)
        for content in CAL_CONTENT:
            rows+=engine.run(content,'mondrian','no_guard',
                             dict(retention=.8,beta_max=.6,gain=2.,fixed_beta=.1),20,
                             out/'reconstruction_diagnostic'/content)
    write_csv(out/'reconstruction_diagnostic.csv',rows)
    for r in rows:
        if r['round'] in [1,20]:
            print(r['content'],r['round'],'SSIM',r['ssim'],'VGG content',r['content_loss'])


if __name__=='__main__': main()
