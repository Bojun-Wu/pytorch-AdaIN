"""Reproducible, CPU-compatible repeated AdaIN experiment (no training)."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import net
from function import adaptive_instance_normalization

ROOT = Path(__file__).resolve().parent
METHODS = ['no_guard', 'low_alpha', 'fixed_blend', 'adaptive_blend',
           'fixed_hf', 'adaptive_hf']
CAL_CONTENT = ['modern', 'avril']
CAL_STYLE = ['mondrian', 'woman_with_hat_matisse']
TEST_CONTENT = ['cornell', 'newyork', 'golden_gate', 'flowers', 'brad_pitt', 'blonde_girl']
TEST_STYLE = ['brushstrokes', 'asheville', 'picasso_self_portrait', 'the_resevoir_at_poitiers']


def load_image(path, size, device):
    # Center crop once; no resize/crop/JPEG re-encoding between rounds.
    im = Image.open(path).convert('RGB')
    w, h = im.size
    scale = size / min(w, h)
    im = im.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    w, h = im.size
    im = im.crop(((w-size)//2, (h-size)//2, (w+size)//2, (h+size)//2))
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div(255)[None].to(device)


def gray(x):
    return (x * x.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)


def blur(x, sigma=1.5):
    radius = int(3*sigma)
    a = torch.arange(-radius, radius+1, device=x.device, dtype=x.dtype)
    k = torch.exp(-a.square()/(2*sigma*sigma)); k /= k.sum()
    kernel = (k[:, None]*k[None, :])[None, None].expand(x.shape[1], 1, -1, -1)
    return F.conv2d(F.pad(x, (radius,)*4, mode='reflect'), kernel, groups=x.shape[1])


def edges(x):
    x = blur(gray(x), 1.0)
    k = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]/8
    p = F.pad(x, (1,)*4, mode='reflect')
    return torch.sqrt(F.conv2d(p, k).square()+F.conv2d(p, k.transpose(2, 3)).square()+1e-12)


def edge_score(reference, x):
    e = edges(x)
    return float((1-(reference-e).abs().sum()/(reference.sum()+e.sum()+1e-8)).clamp(0, 1))


def ssim(x, y):
    x, y = gray(x), gray(y)
    mx, my = blur(x), blur(y)
    vx, vy = blur(x*x)-mx*mx, blur(y*y)-my*my
    cov = blur(x*y)-mx*my
    score = ((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return float(score[..., 5:-5, 5:-5].mean())


def gram(f):
    z = f.flatten(2)
    return z @ z.transpose(1, 2) / z.shape[-1]


def save_tensor(x, path):
    a = (x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()*255).round().astype('uint8')
    Image.fromarray(a).save(path)


class Engine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.vgg = net.vgg.eval()
        self.vgg.load_state_dict(torch.load(ROOT/'models/vgg_normalised.pth', map_location='cpu', weights_only=True))
        self.vgg = self.vgg[:31].to(self.device)
        self.decoder = net.decoder.eval()
        self.decoder.load_state_dict(torch.load(ROOT/'models/decoder.pth', map_location='cpu', weights_only=True))
        self.decoder.to(self.device)
        self.styles = {}
        self.images = {}

    def image(self, kind, name):
        key = (kind, name)
        if key not in self.images:
            paths = list((ROOT/'input'/kind).glob(name+'.*'))
            assert len(paths) == 1, key
            self.images[key] = load_image(paths[0], self.args.size, self.device)
        return self.images[key]

    def features(self, x):
        fs = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in (3, 10, 17, 30): fs.append(x)
        return fs

    def style(self, name):
        if name not in self.styles:
            fs = self.features(self.image('style', name))
            self.styles[name] = (fs[-1], [gram(f) for f in fs])
        return self.styles[name]

    def stylize(self, x, name, alpha):
        f = self.vgg(x)
        sf, _ = self.style(name)
        return self.decoder(alpha*adaptive_instance_normalization(f, sf)+(1-alpha)*f).clamp(0, 1)

    def run(self, content, style, method, params, rounds, directory=None):
        """Repeatedly apply one style to the previous round's corrected output."""
        original = self.image('content', content)
        original_f = self.vgg(original)
        original_e = edges(original)
        original_h = gray(original)-blur(gray(original))
        x = original.clone()
        rows = []
        threshold = None
        total_time = 0
        snapshots = {1, 5, 10, rounds}
        for t in range(1, rounds+1):
            started = time.perf_counter()
            alpha = self.args.alpha * (.5 if method == 'low_alpha' else 1)
            y = self.stylize(x, style, alpha)
            q = edge_score(original_e, y)
            if threshold is None: threshold = params['retention']*q
            beta = 0.
            if t > 1:
                if method.startswith('fixed_'): beta = params['fixed_beta']
                if method.startswith('adaptive_'):
                    beta = min(params['beta_max'], max(0., params['gain']*(threshold-q)/max(threshold, 1e-6)))
            if method.endswith('_blend'):
                x = (1-beta)*y+beta*original
            elif method.endswith('_hf'):
                x = (y+beta*(original_h-(gray(y)-blur(gray(y))))).clamp(0, 1)
            else: x = y
            if self.device.type == 'cuda': torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            total_time += elapsed
            row = dict(content=content, style=style,
                       method=method, round=t, alpha=alpha, beta=beta, triggered=int(beta>0),
                       candidate_edge=q, threshold=threshold, seconds=elapsed, cumulative_seconds=total_time)
            if t in snapshots:
                fs = self.features(x)
                _, gs = self.style(style)
                row.update(ssim=ssim(original, x), edge=edge_score(original_e, x),
                           content_loss=float((fs[-1]-original_f).square().mean()),
                           style_loss=float(torch.stack([(gram(f)-g).square().mean() for f, g in zip(fs, gs)]).mean()))
                if directory:
                    directory.mkdir(parents=True, exist_ok=True)
                    save_tensor(x, directory/f'round_{t:02d}.png')
            rows.append(row)
        return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', help='Output directory (smoke tests use a separate default)')
    p.add_argument('--size', type=int, default=192)
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--alpha', type=float, default=.75)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.output = args.output or ('guard_experiment_smoke' if args.smoke else 'guard_experiment')
    if args.size < 32 or args.size % 8:
        p.error('--size must be a multiple of 8 and at least 32')
    if args.rounds < 1 or args.threads < 1 or not 0 <= args.alpha <= 1:
        p.error('rounds/threads must be positive and alpha must be in [0, 1]')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0); np.random.seed(0)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    manifest = dict(arguments=vars(args), torch=torch.__version__, python=sys.version,
                    platform=platform.platform(), calibration_content=CAL_CONTENT, calibration_style=CAL_STYLE,
                    test_content=TEST_CONTENT, test_style=TEST_STYLE, methods=METHODS,
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    checkpoints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'models').glob('*.pth')})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with torch.inference_mode():
        engine = Engine(args)
        # Calibration selects retention using a content/style-balanced log ratio
        # against no-guard. No test content or style participates in selection.
        base = dict(retention=.8, beta_max=.6, gain=2., fixed_beta=.1)
        if args.smoke:
            rows = []
            for m in METHODS: rows += engine.run('cornell', 'brushstrokes', m, base, 3, out/m)
            write_csv(out/'smoke.csv', rows); print('Smoke complete', flush=True); return
        calibration = []
        scores = []
        for retention in [.6, .8, .95]:
            params = dict(base, retention=retention)
            vals = []
            for c in CAL_CONTENT:
                for s in CAL_STYLE:
                    baseline = engine.run(c, s, 'no_guard', params, 10)[-1]
                    for method in ['adaptive_blend', 'adaptive_hf']:
                        rr = engine.run(c, s, method, params, 10)
                        for row in rr: row['retention'] = retention
                        calibration += rr
                        r = rr[-1]
                        vals.append(np.log((r['content_loss']+1e-8)/(baseline['content_loss']+1e-8))+
                                    np.log((r['style_loss']+1e-8)/(baseline['style_loss']+1e-8)))
            score = float(np.mean(vals)); scores.append((score, retention))
            print('Calibration', retention, score, flush=True)
        selected = dict(base, retention=min(scores)[1])
        write_csv(out/'calibration.csv', calibration)
        (out/'parameters.json').write_text(json.dumps(dict(selected=selected, selection_scores=scores,
            objective='mean(log(content_loss/no_guard)+log(style_loss/no_guard)), both adaptive methods, round 10'), indent=2))
        print('Selected', selected, flush=True)
        rows = []
        for ci, c in enumerate(TEST_CONTENT):
            for si, s in enumerate(TEST_STYLE):
                for m in METHODS:
                    folder = out/'images'/f'{c}__{s}'/m
                    rows += engine.run(c, s, m, selected, args.rounds, folder)
                    # Checkpoint complete sequences: interruption preserves completed data.
                    write_csv(out/'metrics.csv', rows)
                pair = ci*len(TEST_STYLE)+si+1
                print(f'Test {pair}/{len(TEST_CONTENT)*len(TEST_STYLE)}: {c} {s}', flush=True)
        print('Experiment complete:', out.resolve(), flush=True)


if __name__ == '__main__': main()
