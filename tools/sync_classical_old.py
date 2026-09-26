import os
import sys
import difflib
import argparse


SEARCH_CONSTANTS_MARKER = '# --- Search Constants / 搜尋常量 ---'


def get_py_files(directory):
    files = []
    for f in os.listdir(directory):
        if f.endswith('.py') and f != '__init__.py':
            files.append(f)
    return sorted(files)

def normalize_imports(content, to_old=False):
    if to_old:
        return content.replace('chess_engine.classical.', 'chess_engine.classical_old.')
    else:
        return content.replace('chess_engine.classical_old.', 'chess_engine.classical.')


def _section_from_marker(content, marker=SEARCH_CONSTANTS_MARKER):
    index = content.find(marker)
    if index < 0:
        raise ValueError(f'Missing constants section marker: {marker}')
    return content[index:]


def _sync_search_constants(src_path, dest_path):
    with open(src_path, 'r', encoding='utf-8') as sf:
        src_content = sf.read()
    with open(dest_path, 'r', encoding='utf-8') as df:
        dest_content = df.read()

    dest_index = dest_content.find(SEARCH_CONSTANTS_MARKER)
    if dest_index < 0:
        raise ValueError(f'Missing constants section marker in {dest_path}')
    updated_content = dest_content[:dest_index] + _section_from_marker(src_content)
    if updated_content == dest_content:
        return False

    with open(dest_path, 'w', encoding='utf-8') as df:
        df.write(updated_content)
    return True


def do_diff(src_dir, dest_dir, ignored_files=None, announce=True):
    ignored_files = set(ignored_files or ())
    src_files = get_py_files(src_dir)
    dest_files = get_py_files(dest_dir)
    
    all_files = sorted(list(set(src_files + dest_files)))
    has_diff = False
    
    for f in all_files:
        if f in ignored_files:
            continue

        src_path = os.path.join(src_dir, f)
        dest_path = os.path.join(dest_dir, f)
        
        if not os.path.exists(src_path):
            print(f"[-] File {f} exists only in classical_old")
            has_diff = True
            continue
        if not os.path.exists(dest_path):
            print(f"[-] File {f} exists only in classical")
            has_diff = True
            continue
            
        with open(src_path, 'r', encoding='utf-8') as sf:
            src_content = sf.read()
        with open(dest_path, 'r', encoding='utf-8') as df:
            dest_content = df.read()
            
        # Normalize both to 'chess_engine.classical' for logic comparison
        src_norm = normalize_imports(src_content, to_old=False).splitlines()
        dest_norm = normalize_imports(dest_content, to_old=False).splitlines()
        
        diff = list(difflib.unified_diff(dest_norm, src_norm, fromfile=f'classical_old/{f}', tofile=f'classical/{f}', lineterm=''))
        if diff:
            print(f"\n[!] Differences found in {f}:")
            for line in diff:
                print(line)
            has_diff = True
            
    if announce and not has_diff:
        print("[+] Both directories are logically identical!")
    return has_diff


def do_diff_code(src_dir, dest_dir):
    """Verify identical engine code and search constants, allowing eval weights to differ."""
    has_diff = do_diff(
        src_dir, dest_dir, ignored_files={"constants.py"}, announce=False
    )
    src_path = os.path.join(src_dir, 'constants.py')
    dest_path = os.path.join(dest_dir, 'constants.py')
    with open(src_path, 'r', encoding='utf-8') as sf:
        src_tail = _section_from_marker(sf.read()).splitlines()
    with open(dest_path, 'r', encoding='utf-8') as df:
        dest_tail = _section_from_marker(df.read()).splitlines()

    diff = list(difflib.unified_diff(
        dest_tail, src_tail,
        fromfile='classical_old/constants.py (search section)',
        tofile='classical/constants.py (search section)',
        lineterm='',
    ))
    if diff:
        print("\n[!] Differences found in search constants:")
        for line in diff:
            print(line)
        has_diff = True

    if not has_diff:
        print("[+] Engine code and search constants are logically identical!")
    return has_diff


def do_sync(src_dir, dest_dir, preserve_constants=False):
    src_files = get_py_files(src_dir)
    print(f"[*] Syncing files from {src_dir} to {dest_dir}...")
    skipped = {"constants.py"} if preserve_constants else set()
    updated = 0
    
    for f in src_files:
        src_path = os.path.join(src_dir, f)
        dest_path = os.path.join(dest_dir, f)

        if f in skipped:
            if _sync_search_constants(src_path, dest_path):
                print("  [+] Preserved evaluation constants; synced search constants")
                updated += 1
            else:
                print("  [=] Preserved evaluation constants; search constants unchanged")
            continue
        
        with open(src_path, 'r', encoding='utf-8') as sf:
            content = sf.read()
            
        # Modify path dependency to point to classical_old
        updated_content = normalize_imports(content, to_old=True)

        if os.path.exists(dest_path):
            with open(dest_path, 'r', encoding='utf-8') as df:
                if df.read() == updated_content:
                    continue

        with open(dest_path, 'w', encoding='utf-8') as df:
            df.write(updated_content)
        print(f"  [+] Synced and updated imports for {f}")
        updated += 1
    
    print(f"[+] Sync complete! Updated {updated} file(s).")

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is not None:
            reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="Sync and Diff chess_engine/classical and chess_engine/classical_old")
    parser.add_argument('--diff', action='store_true', help="Show logical differences between directories (ignores import paths)")
    parser.add_argument(
        '--diff-code', action='store_true',
        help="Compare engine code while ignoring the intentional constants.py baseline",
    )
    parser.add_argument('--sync', action='store_true', help="Fully copy classical files to classical_old, updating import dependencies")
    parser.add_argument(
        '--sync-code', action='store_true',
        help="Copy engine code while preserving classical_old/constants.py as the battle baseline",
    )
    
    args = parser.parse_args()
    
    # Absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, 'chess_engine', 'classical')
    dest_dir = os.path.join(base_dir, 'chess_engine', 'classical_old')
    
    selected = sum(bool(value) for value in (args.diff, args.diff_code, args.sync, args.sync_code))
    if selected > 1:
        parser.error('--diff, --diff-code, --sync and --sync-code are mutually exclusive')

    if args.sync:
        do_sync(src_dir, dest_dir)
    elif args.sync_code:
        do_sync(src_dir, dest_dir, preserve_constants=True)
    elif args.diff_code:
        do_diff_code(src_dir, dest_dir)
    else:
        # Default to diff when --diff is specified or no options are provided
        do_diff(src_dir, dest_dir)
