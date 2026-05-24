"""
Extracts all image outputs from all phase notebooks and saves as PNG files.
Output: plots/<phase>/<cell_index>.png
Run: python extract_all_plots.py
"""
import json, base64, os

NOTEBOOKS = [
    ('phase1', 'notebooks/phase1_sentiment140.ipynb'),
    ('phase2_naive_bayes',        'notebooks/phase2_naive_bayes.ipynb'),
    ('phase2_logistic_regression','notebooks/phase2_logistic_regression.ipynb'),
    ('phase2_linear_svc',         'notebooks/phase2_linear_svc.ipynb'),
    ('phase2_random_forest',      'notebooks/phase2_random_forest.ipynb'),
    ('phase3', 'notebooks/phase3_evaluation.ipynb'),
    ('phase4', 'notebooks/phase4_error_analysis.ipynb'),
    ('phase5', 'notebooks/phase5_improvements.ipynb'),
    ('phase6', 'notebooks/phase6_bert_colab.ipynb'),
]

total = 0
for phase_name, nb_path in NOTEBOOKS:
    if not os.path.exists(nb_path):
        print(f"SKIP (not found): {nb_path}")
        continue

    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    out_dir = f'plots/{phase_name}'
    os.makedirs(out_dir, exist_ok=True)
    count = 0

    for cell_idx, cell in enumerate(nb['cells']):
        for out_idx, output in enumerate(cell.get('outputs', [])):
            data = output.get('data', {})
            img_b64 = data.get('image/png')
            if not img_b64:
                continue
            if isinstance(img_b64, list):
                img_b64 = ''.join(img_b64)
            fname = f'{out_dir}/cell{cell_idx:02d}_{out_idx}.png'
            with open(fname, 'wb') as f:
                f.write(base64.b64decode(img_b64))
            count += 1
            print(f"  {fname}")

    print(f"{phase_name}: {count} images saved")
    total += count

print(f"\nTotal: {total} images saved to plots/")