import re

with open('tune_search/search_param_registry.py', 'r', encoding='utf-8') as f:
    text = f.read()

print("--- Parameters in tune_search/search_param_registry.py ---")
for line in text.splitlines():
    if '_spec(' in line:
        print(line.strip())

