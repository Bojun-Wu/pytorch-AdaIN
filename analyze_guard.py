"""Validate fixed-style measurements and regenerate reports without inference."""
import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
LABELS = {
    'no_guard': 'No guard', 'low_alpha': 'Lower alpha',
    'fixed_blend': 'Fixed RGB blend', 'adaptive_blend': 'Adaptive RGB blend',
    'fixed_hf': 'Fixed HF repair', 'adaptive_hf': 'Adaptive HF repair',
}
COLORS = dict(zip(LABELS, ['#64748b', '#a16207', '#059669', '#2563eb', '#b45309', '#dc2626']))
METRICS = ['ssim', 'edge', 'content_loss', 'style_loss']


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ['round', 'beta', 'triggered', 'seconds', *METRICS]:
            row[key] = float(row[key]) if row.get(key) else None
    return rows


def mean(rows, key):
    return float(np.mean([row[key] for row in rows if row[key] is not None]))


def chart(path, series, title):
    xs = [x for points in series.values() for x, _ in points]
    ys = [y for points in series.values() for _, y in points]
    ymin, ymax = min(0, min(ys)), max(ys)*1.08 or 1
    xx = lambda x: 90+(x-min(xs))/(max(xs)-min(xs) or 1)*650
    yy = lambda y: 470-(y-ymin)/(ymax-ymin)*400
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="560" viewBox="0 0 1020 560">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,sans-serif" font-size="14" fill="#1e293b">',
             f'<text x="40" y="32" font-size="21">{html.escape(title)}</text>']
    for y in np.linspace(ymin, ymax, 6):
        parts += [f'<path d="M 90 {yy(y)} H 740" stroke="#e2e8f0"/>',
                  f'<text x="80" y="{yy(y)+5}" text-anchor="end">{y:.3g}</text>']
    for x in sorted(set(xs)):
        parts.append(f'<text x="{xx(x)}" y="495" text-anchor="middle">{x:g}</text>')
    for i, (method, points) in enumerate(series.items()):
        color = COLORS[method]
        coordinates = ' '.join(f'{xx(x)},{yy(y)}' for x, y in points)
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            parts.append(f'<circle cx="{xx(x)}" cy="{yy(y)}" r="4" fill="{color}"/>')
        parts += [f'<rect x="775" y="{75+i*30}" width="16" height="4" fill="{color}"/>',
                  f'<text x="800" y="{82+i*30}">{LABELS[method]}</text>']
    parts += ['<text x="415" y="535" text-anchor="middle">Stylization round</text>', '</g></svg>']
    path.write_text('\n'.join(parts), encoding='utf-8')


def font():
    for name in ['DejaVuSans.ttf', 'Arial.ttf', 'C:/Windows/Fonts/arial.ttf']:
        try:
            return ImageFont.truetype(name, 16)
        except OSError:
            pass
    return ImageFont.load_default(size=16)


def thumb(kind, name, size):
    path = next((ROOT/'input'/kind).glob(name+'.*'))
    with Image.open(path) as image:
        image = image.convert('RGB')
        w, h = image.size
        scale = size/min(w, h)
        image = image.resize((round(w*scale), round(h*scale)), Image.Resampling.LANCZOS)
        w, h = image.size
        return image.crop(((w-size)//2, (h-size)//2, (w+size)//2, (h+size)//2))


def examples(out, manifest, snapshots):
    """Select one example per photo by list order, never by score."""
    size = manifest['arguments']['size']
    cell, height = max(200, size+8), size+30
    names = []
    for i, content in enumerate(manifest['test_content']):
        style = manifest['test_style'][i % len(manifest['test_style'])]
        name = f'example_{i+1}_{content}.jpg'
        names.append(name)
        if not (out/'images').exists():
            continue  # A published checkout retains its existing contact sheets.
        canvas = Image.new('RGB', ((len(snapshots)+1)*cell, (len(LABELS)+1)*height), '#fafafa')
        draw = ImageDraw.Draw(canvas)
        canvas.paste(thumb('content', content, size), (4, 24))
        canvas.paste(thumb('style', style, size), (cell+4, 24))
        draw.text((4, 3), 'Original: '+content, fill='black', font=font())
        draw.text((cell+4, 3), 'Style reference', fill='black', font=font())
        for j, method in enumerate(LABELS):
            y = (j+1)*height
            draw.text((4, y+70), LABELS[method], fill='black', font=font())
            for k, t in enumerate(snapshots):
                path = out/'images'/f'{content}__{style}'/method/f'round_{t:02d}.png'
                with Image.open(path) as image:
                    canvas.paste(image, ((k+1)*cell+4, y+24))
                draw.text(((k+1)*cell+4, y+3), f'Round {t}', fill='black', font=font())
        canvas.save(out/name, quality=93)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='guard_experiment')
    parser.add_argument('--require-images', action='store_true', help='Require all local raw PNG checkpoints')
    args = parser.parse_args()
    out = Path(args.output)
    rows = read(out/'metrics.csv')
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    params = json.loads((out/'parameters.json').read_text(encoding='utf-8'))
    rounds = manifest['arguments']['rounds']
    contents, styles, methods = (manifest[k] for k in ['test_content', 'test_style', 'methods'])
    snapshots = sorted({t for t in [1, 5, 10, rounds] if t <= rounds})
    assert methods == list(LABELS), 'Unexpected method set'
    expected = {(c, s, m, t) for c in contents for s in styles for m in methods for t in range(1, rounds+1)}
    keys = [(r['content'], r['style'], r['method'], r['round']) for r in rows]
    assert len(keys) == len(set(keys)) and set(keys) == expected, 'Incomplete or duplicate measurements'
    assert all(np.isfinite(r[k]) for r in rows for k in ['seconds', 'beta'])
    assert all(np.isfinite(r[k]) for r in rows if r['round'] in snapshots for k in METRICS)
    assert set(manifest['calibration_content']).isdisjoint(contents)
    assert set(manifest['calibration_style']).isdisjoint(styles)
    image_paths = {out/'images'/f'{c}__{s}'/m/f'round_{t:02d}.png' for c in contents for s in styles for m in methods for t in snapshots}
    raw_images = set((out/'images').rglob('*.png'))
    if args.require_images or (out/'images').exists():
        assert raw_images == image_paths, 'Missing or unexpected PNG checkpoints'
        for path in raw_images:
            with Image.open(path) as image:
                assert image.size == (manifest['arguments']['size'],)*2
    validation = dict(complete=True, rows=len(rows), sequences=len(rows)//rounds,
                      expected_png_checkpoints=len(image_paths), raw_images_verified=bool(raw_images),
                      unique_sequence_rounds=True, finite_metrics=True, disjoint_photos_and_styles=True,
                      analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    (out/'analysis_environment.json').write_text(json.dumps({k: importlib.metadata.version(k) for k in ['numpy', 'Pillow']}, indent=2), encoding='utf-8')
    final = [r for r in rows if r['round'] == rounds]
    summary = []
    table = ['| Method | Gray SSIM ↑ | Edge score ↑ | VGG content ↓ | Gram style ↓ | ms/round* | Trigger % |', '|---|---:|---:|---:|---:|---:|---:|']
    for method in methods:
        rr = [r for r in final if r['method'] == method]
        allr = [r for r in rows if r['method'] == method]
        summary.append(dict(method=method, **{k: mean(rr, k) for k in METRICS}))
        table.append(f"| {LABELS[method]} | {mean(rr,'ssim'):.4f} | {mean(rr,'edge'):.4f} | {mean(rr,'content_loss'):.4f} | {mean(rr,'style_loss'):.4f} | {1000*mean(allr,'seconds'):.1f} | {100*mean(allr,'triggered'):.1f} |")
    with (out/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for metric in METRICS:
        series = {m: [(t, mean([r for r in rows if r['method'] == m and r['round'] == t], metric)) for t in snapshots] for m in methods}
        chart(out/f'{metric}.svg', series, f'Fixed-style repeated AdaIN: {metric}')
    series = {m: [(t, mean([r for r in rows if r['method'] == m and r['round'] == t], 'beta')) for t in range(1, rounds+1)] for m in methods if m not in ['no_guard', 'low_alpha']}
    chart(out/'interventions.svg', series, 'Mean restoration weight beta')
    baseline = {(r['content'], r['style']): r for r in final if r['method'] == 'no_guard'}
    paired, rng = [], np.random.default_rng(0)
    for method in methods:
        for metric in ['content_loss', 'style_loss']:
            # Average pairwise ratios within photo before bootstrapping photo clusters.
            values = np.array([np.mean([r[metric]/max(baseline[(r['content'], r['style'])][metric], 1e-8) for r in final if r['method'] == method and r['content'] == c]) for c in contents])
            boot = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(1)
            paired.append(dict(method=method, metric=metric, ratio=float(values.mean()), ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975))))
    (out/'paired_bootstrap.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
    paired_table = ['| Method | Content ratio [95% interval] | Style ratio [95% interval] |', '|---|---:|---:|']
    for method in methods:
        a, b = [next(r for r in paired if r['method'] == method and r['metric'] == k) for k in ['content_loss', 'style_loss']]
        paired_table.append(f"| {LABELS[method]} | {a['ratio']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}] | {b['ratio']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}] |")
    example_names = examples(out, manifest, snapshots)
    lines = ['# Adaptive Content Guard: experimental results', '\n## Scope',
             f"\n{len(contents)} test photos × {len(styles)} fixed styles × {len(methods)} methods × {rounds} rounds = {len(rows):,} outputs. Each sequence repeatedly applies one style to the previous round. Resolution: {manifest['arguments']['size']} square pixels; primary alpha: {manifest['arguments']['alpha']}. Calibration photos and styles are disjoint from test assets. No model training was performed.",
             '\nSelected parameters: `'+json.dumps(params['selected'])+'`.',
             f'\n## Round-{rounds} results\n\n'+'\n'.join(table),
             '\n*Timing includes inference, edge monitoring and correction; it excludes evaluation, saving and model loading. All methods compute monitoring for diagnostics. Single-run CPU timings describe this harness, not optimized standalone inference.*',
             '\n## Paired loss ratios\n\n'+'\n'.join(paired_table),
             '\nRatios compare the same photo/style pair against no guard at the final round. Below 1 means lower error. Ratios are averaged within photo, then across photos. Intervals use 2,000 photo-cluster bootstrap samples, seed 0. Six photos provide limited generalizability.',
             '\n## Interpretation',
             '\nLower content loss with higher style loss represents a trade-off, not an unconditional improvement. The adaptive HF guard does not establish superiority over fixed HF repair here. Edge monitoring and grayscale residual repair cannot guarantee semantic or geometric preservation; examples still show structural drift.',
             '\n## Metrics and limitations',
             '\nContent loss is VGG relu4_1 feature MSE against the original. Style loss averages Gram MSE at relu1_1/2_1/3_1/4_1 against the fixed style, with Gram=FFᵀ/(HW). Grayscale SSIM uses Gaussian sigma 1.5, C1=0.01², C2=0.03² and a five-pixel excluded border. Edge score is the controller signal, not independent evidence. Raw loss values are implementation-specific; a reduction is not a percentage of semantic content retained.',
             '\nOnly AdaIN, one primary alpha and one resolution were tested. Fixed beta was not exhaustively tuned. High-frequency repair can erase artistic texture, cause halos and clipping. No human aesthetic study or broad novelty claim is established. See [README_GUARD.md](../README_GUARD.md) for the method, citations and reproduction commands.']
    diagnostic_path = out/'reconstruction_diagnostic.csv'
    if diagnostic_path.exists():
        lines += ['\n## Alpha-zero reconstruction diagnostic', '\nTwo calibration photos each traverse the encoder/decoder for 20 rounds at alpha=0. These 40 outputs are separate from the main experiment.', '\n| Photo | Round | SSIM | VGG content loss |\n|---|---:|---:|---:|']
        for row in read(diagnostic_path):
            if row['round'] in [1, 20]:
                lines.append(f"| {row['content']} | {row['round']:.0f} | {row['ssim']:.4f} | {row['content_loss']:.4f} |")
    lines.append('\n## Figures')
    for name in [*METRICS, 'interventions']:
        lines.append(f'\n![{name}]({name}.svg)')
    for name in example_names:
        lines.append(f'\n![Predetermined qualitative example]({name})')
    lines.append('\nExamples are chosen by list order, without selecting favorable scores. Full PNG checkpoints are local outputs excluded from Git. Compact CSV/JSON results, SVG plots and JPEG contact sheets are published.')
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    zh = ['# Adaptive Content Guard 實驗結果', f'\n保留 {len(contents)*len(styles)} 組固定風格配對、{len(methods)} 種方法、每組 {rounds} 輪，共 {len(rows):,} 筆逐輪結果。原始推論未重新執行；統計與圖表依保留資料重新產生。', '\n第 20 輪相對無 guard 的配對平均 loss 比值如下，兩者皆愈低愈好。比值不代表語意內容的保留百分比。', '\n'+'\n'.join(paired_table), '\n目前資料沒有支持自適應高頻修復優於固定高頻修復。必須同時考慮內容與風格誤差。', '\n[完整英文結果](RESULTS.md) · [英文專案說明](../README_GUARD.md)']
    (out/'FINDINGS_zh.md').write_text('\n'.join(zh), encoding='utf-8')
    print(f'Validated {len(rows)} rows, {len(final)} sequences and {len(raw_images)} local PNG checkpoints.')
    print('\n'.join(table))
    print('\n'.join(paired_table))


if __name__ == '__main__':
    main()
