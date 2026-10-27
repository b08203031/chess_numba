with open('chess_engine/classical/constants.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re
# Find all UPPER_CASE variable assignments
assigns = re.findall(r'^[A-Z0-9_]+\s*=\s*.*$', text, re.MULTILINE)
print(f"Total uppercase assignments: {len(assigns)}")

# Filter to search-related sections
search_marker = '# --- Search Constants / 搜尋常量 ---'
idx = text.find(search_marker)
if idx != -1:
    search_text = text[idx:]
    search_assigns = re.findall(r'^([A-Z0-9_]+)\s*=\s*(.*?)(?:#.*)?$', search_text, re.MULTILINE)
    with open('scratch/search_constants_list.txt', 'w', encoding='utf-8') as out:
        for name, val in search_assigns:
            if not name.startswith('DIAG_') and not name.startswith('TUNE_'):
                out.write(f"{name:35} = {val.strip()}\n")
    print(f"Wrote search constants.")

