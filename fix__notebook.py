import json, hashlib

with open('DDPG.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['nbformat_minor'] = 5

nb['metadata']['widgets'] = {
    "application/vnd.jupyter.widget-state+json": {
        "state": {},
        "version_major": 2,
        "version_minor": 0
    }
}

for i, cell in enumerate(nb['cells']):
    if 'id' not in cell:
        cell['id'] = hashlib.md5(f'cell-{i}'.encode()).hexdigest()[:8]

with open('DDPG.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("완료!")