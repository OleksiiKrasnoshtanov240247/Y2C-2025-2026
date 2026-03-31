import json

nb_path = r'c:\Users\Alex\Documents\GitHub\Y2C-2025-2026\oleksii_krasnoshtanov_eda.ipynb'
out_path = r'c:\Users\Alex\Documents\GitHub\Y2C-2025-2026\nb_structure.md'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(out_path, 'w', encoding='utf-8') as f:
    for i, cell in enumerate(nb.get('cells', [])):
        f.write(f"\n\n--- Cell {i} ({cell['cell_type']}) ---\n")
        source = cell.get('source', [])
        if isinstance(source, list):
            f.write("".join(source))
        else:
            f.write(source)
print("Extracted notebook structure to nb_structure.md")
