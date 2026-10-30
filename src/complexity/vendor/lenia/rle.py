# Ported from the original Lenia (~/github/lenia-ref, commit
# adfc542939266de7f4bb7ebb552e8499701ee107), code/Lenia/Python/LeniaND.py: `Board.ch2val`
# verbatim, `Board.rle2arr` reduced to 2-D (DIM = 2, row delimiter `$`).
"""Decode Lenia RLE: cell values 0..255 as `.`/`b` (0), `o` (255), `A`..`X` (1..24) or a
prefix `p`..`y` plus `A`..`X` (25..254); runs are prefixed by a count; `$` ends a row and
`!` the pattern. Plain Life RLE (`b`, `o`) is a special case."""

import numpy as np


def ch2val(c):
    if c in '.b': return 0
    elif c == 'o': return 255
    elif len(c) == 1: return ord(c)-ord('A')+1
    else: return (ord(c[0])-ord('p')) * 24 + (ord(c[1])-ord('A')+25)


def rle2arr(st: str) -> np.ndarray:
    """float32 (height, width) in [0, 1]; short rows are padded with zeros."""
    rows, row = [], []
    last, count = '', ''
    for ch in st.rstrip('!') + '$':
        if ch.isdigit(): count += ch
        elif ch in 'pqrstuvwxy@': last = ch
        elif ch == '$':
            rows.append(row)
            rows.extend([[] for _ in range(int(count or 1) - 1)])
            row, last, count = [], '', ''
        else:
            row.extend([ch2val(last + ch) / 255] * int(count or 1))
            last, count = '', ''
    width = max(len(r) for r in rows)
    return np.array([r + [0] * (width - len(r)) for r in rows], dtype=np.float32)
