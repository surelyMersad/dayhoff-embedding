import requests
import random

# Fetch and save protein-only PDB IDs
url = "https://files.rcsb.org/pub/pdb/derived_data/pdb_entry_type.txt"
lines = requests.get(url).text.split('\n')[1:]
# choose 5000 random protein PDB IDs
pdb_ids = random.sample([line.split()[0].lower() for line in lines if line.strip() and line.split()[1] == 'prot'], 5000)

# Save to pdb_ids.py
with open('pdb_ids.py', 'w') as f:
    f.write('pdb_list = [\n')
    for i in range(0, len(pdb_ids), 10):
        chunk = ', '.join(f'"{p}"' for p in pdb_ids[i:i+10])
        f.write(f'    {chunk},\n')
    f.write(']\n')

print(f"Saved {len(pdb_ids)} IDs to pdb_ids.py")
