import re

path = r'c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba\tests\perft.py'
with open(path, 'rb') as f:
    content = f.read()

# Decode as latin-1 to keep bytes exactly
text = content.decode('latin-1')

# Replace docstrings for each function to fix syntax issues
text = re.sub(
    r'(def do_perft\(args\):[ \t\r\n]+)""".*?""',
    r'\1"""Execute perft command"""',
    text
)

text = re.sub(
    r'(def do_divide\(args\):[ \t\r\n]+)""".*?""',
    r'\1"""Execute divide command"""',
    text
)

text = re.sub(
    r'(def do_test_reversibility\(args\):[ \t\r\n]+)""".*?""',
    r'\1"""Execute test-reversibility command"""',
    text
)

text = re.sub(
    r'(def do_run_tests\(args\):[ \t\r\n]+)""".*?""',
    r'\1"""Execute run-tests command"""',
    text
)

text = re.sub(
    r'(def get_test_key\(fen: str\) -> str:[ \t\r\n]+)""".*?""',
    r'\1"""Get expected perft result key for FEN"""',
    text
)

text = re.sub(
    r'(def test_perft_suite\(\):[ \t\r\n]+)""".*?""',
    r'\1"""Run a series of perft tests to verify move generator"""',
    text
)

text = re.sub(
    r'(def run_perft_test\(fen_string: str, max_depth: int, divide_on_mismatch: bool = False\):[ \t\r\n]+)""".*?"""',
    r'\1"""Run perft test for given FEN string"""',
    text,
    flags=re.DOTALL
)

text = re.sub(
    r'(def perft_divide\(fen_string: str, depth: int\):[ \t\r\n]+)""".*?""',
    r'\1"""Show node count for each move at given depth"""',
    text
)

text = re.sub(
    r'(def test_make_unmake_for_fen\(fen: str\):[ \t\r\n]+)""".*?""',
    r'\1"""Test if make_move and unmake_move are completely reversible"""',
    text
)

# Sanitize all remaining non-ASCII characters to standard ASCII '?'
text = re.sub(r'[^\x00-\x7F]+', '?', text)

# Save as UTF-8
with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("tests/perft.py successfully sanitized and docstrings fixed.")
