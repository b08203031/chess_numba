import csv
from collections import defaultdict

data = defaultdict(list)
with open('artifacts/calibration_p1_results.csv', mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        mode = row['mode']
        id_ = row['id']
        bucket = row['bucket']
        try:
            d6_nodes = int(row['d6_nodes'])
        except (ValueError, TypeError):
            d6_nodes = None
        if d6_nodes is not None:
            data[mode].append((id_, bucket, d6_nodes))

print("Mode | Average d6_nodes | Median d6_nodes")
print("---|---|---")
for mode in sorted(data.keys()):
    nodes = [item[2] for item in data[mode]]
    if nodes:
        avg_nodes = sum(nodes) / len(nodes)
        sorted_nodes = sorted(nodes)
        mid = len(sorted_nodes) // 2
        med_nodes = sorted_nodes[mid] if len(sorted_nodes) % 2 != 0 else (sorted_nodes[mid-1] + sorted_nodes[mid]) / 2
        print(f"{mode} | {avg_nodes:.1f} | {med_nodes:.1f}")

print("\nDetail of nodes per position (Mode: full_search vs no_lmr vs no_nmp_rfp_razor vs no_see_pruning):")
pos_modes = defaultdict(dict)
for mode, items in data.items():
    for id_, bucket, d6_nodes in items:
        pos_modes[id_][mode] = d6_nodes

print("ID | Bucket | full_search | no_lmr | no_nmp_rfp_razor | no_see_pruning")
print("---|---|---|---|---|---")
for id_ in sorted(pos_modes.keys()):
    modes_dict = pos_modes[id_]
    bucket = [item[1] for items in data.values() for item in items if item[0] == id_][0]
    print(f"{id_} | {bucket} | {modes_dict.get('full_search', '-')} | {modes_dict.get('no_lmr', '-')} | {modes_dict.get('no_nmp_rfp_razor', '-')} | {modes_dict.get('no_see_pruning', '-')}")
