"""
Injects phase3_plots PNGs as outputs into the 4 empty cells in phase3_evaluation.ipynb.
Run once — no retraining needed.
"""
import json, base64, copy

NB_PATH   = 'notebooks/phase3_evaluation.ipynb'
PLOTS_DIR = 'phase3_plots'

# Map: substring to find in cell source -> PNG file to inject
CELL_MAP = {
    'roc_curve':                 f'{PLOTS_DIR}/roc_curves.png',
    'precision_recall_curve':    f'{PLOTS_DIR}/pr_curves.png',
    'cross_val_score':           f'{PLOTS_DIR}/cv_results.png',
    'learning_curve':            f'{PLOTS_DIR}/learning_curve.png',
}

def make_image_output(png_path):
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {
            "image/png": b64,
            "text/plain": ["<Figure>"]
        }
    }

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

injected = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    for keyword, png_path in CELL_MAP.items():
        if keyword in src and not cell.get('outputs'):
            cell['outputs'] = [make_image_output(png_path)]
            cell['execution_count'] = 1
            print(f"  Injected {png_path} -> cell containing '{keyword}'")
            injected += 1
            break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nDone — {injected} cells updated in {NB_PATH}")