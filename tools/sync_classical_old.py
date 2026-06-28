import os
import sys
import difflib
import argparse

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

def do_diff(src_dir, dest_dir):
    src_files = get_py_files(src_dir)
    dest_files = get_py_files(dest_dir)
    
    all_files = sorted(list(set(src_files + dest_files)))
    has_diff = False
    
    for f in all_files:
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
            
    if not has_diff:
        print("[+] Both directories are logically identical!")
    return has_diff

def do_sync(src_dir, dest_dir):
    src_files = get_py_files(src_dir)
    print(f"[*] Syncing files from {src_dir} to {dest_dir}...")
    
    for f in src_files:
        src_path = os.path.join(src_dir, f)
        dest_path = os.path.join(dest_dir, f)
        
        with open(src_path, 'r', encoding='utf-8') as sf:
            content = sf.read()
            
        # Modify path dependency to point to classical_old
        updated_content = normalize_imports(content, to_old=True)
        
        with open(dest_path, 'w', encoding='utf-8') as df:
            df.write(updated_content)
        print(f"  [+] Synced and updated imports for {f}")
        
    print("[+] Sync complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sync and Diff chess_engine/classical and chess_engine/classical_old")
    parser.add_argument('--diff', action='store_true', help="Show logical differences between directories (ignores import paths)")
    parser.add_argument('--sync', action='store_true', help="Fully copy classical files to classical_old, updating import dependencies")
    
    args = parser.parse_args()
    
    # Absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, 'chess_engine', 'classical')
    dest_dir = os.path.join(base_dir, 'chess_engine', 'classical_old')
    
    if args.diff:
        do_diff(src_dir, dest_dir)
    elif args.sync:
        do_sync(src_dir, dest_dir)
    else:
        parser.print_help()
        sys.exit(1)
