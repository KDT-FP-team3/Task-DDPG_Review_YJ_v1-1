import json

notebook_path = "DDPG.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check if widgets metadata exists
if 'widgets' in nb.get('metadata', {}):
    print("Found 'widgets' in metadata. Removing it to fix rendering error.")
    del nb['metadata']['widgets']
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully removed widgets metadata and saved the notebook.")
else:
    print("'widgets' key not found in metadata. Checking cells...")
    # Sometimes it's in the outputs of cells
    changed = False
    for cell in nb.get('cells', []):
        if 'metadata' in cell and 'widgets' in cell['metadata']:
            print(f"Found 'widgets' in cell metadata at line {cell.get('execution_count')}. Removing...")
            del cell['metadata']['widgets']
            changed = True
    
    if changed:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Successfully cleaned cell metadata and saved the notebook.")
    else:
        print("No 'widgets' metadata found in cells either.")
