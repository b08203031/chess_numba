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
            d6_nodes = int(row['d6_nodes']) if row.get('d6_nodes') else None
        except (ValueError, TypeError):
            d6_nodes = None
        try:
            d8_nodes = int(row['d8_nodes']) if row.get('d8_nodes') else None
        except (ValueError, TypeError):
            d8_nodes = None
        try:
            d10_nodes = int(row['d10_nodes']) if row.get('d10_nodes') else None
        except (ValueError, TypeError):
            d10_nodes = None
        data[mode].append((id_, bucket, d6_nodes, d8_nodes, d10_nodes))

def print_stats(depth_name, idx):
    print(f"\n### Stats for {depth_name}")
    print("Mode | Average nodes | Median nodes")
    print("---|---|---")
    for mode in sorted(data.keys()):
        nodes = [item[idx] for item in data[mode] if item[idx] is not None]
        if nodes:
            avg_nodes = sum(nodes) / len(nodes)
            sorted_nodes = sorted(nodes)
            mid = len(sorted_nodes) // 2
            med_nodes = sorted_nodes[mid] if len(sorted_nodes) % 2 != 0 else (sorted_nodes[mid-1] + sorted_nodes[mid]) / 2
            print(f"{mode} | {avg_nodes:.1f} | {med_nodes:.1f}")

print_stats("depth 6", 2)
print_stats("depth 8", 3)
print_stats("depth 10", 4)

print("\nDetail of depth 10 nodes per position (Mode: full_search vs no_lmr vs no_nmp_rfp_razor vs no_see_pruning):")
pos_modes = defaultdict(dict)
for mode, items in data.items():
    for id_, bucket, d6_nodes, d8_nodes, d10_nodes in items:
        if d10_nodes is not None:
            pos_modes[id_][mode] = d10_nodes

print("ID | Bucket | full_search | no_lmr | no_nmp_rfp_razor | no_see_pruning")
print("---|---|---|---|---|---")
for id_ in sorted(pos_modes.keys()):
    modes_dict = pos_modes[id_]
    bucket = [item[1] for items in data.values() for item in items if item[0] == id_][0]
    print(f"{id_} | {bucket} | {modes_dict.get('full_search', '-')} | {modes_dict.get('no_lmr', '-')} | {modes_dict.get('no_nmp_rfp_razor', '-')} | {modes_dict.get('no_see_pruning', '-')}")

