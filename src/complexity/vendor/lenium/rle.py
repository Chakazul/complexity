# Vendored from Lenium (~/github/lenium), src/lenium/io/rle.py at commit 139492ce0227ec2556f6455901f681676cfaace7.
# Copied verbatim; replace with `import lenium` once Lenium is published (PLAN.md §4.2).
"""RLE codec for creature cells — an nD extension of Golly's run-length format.

Ported from the predecessor's `io.rle2arr` / `io.arr2rle`, which in turn extend
`https://golly.sourceforge.io/Help/formats.html` past two dimensions.

Alphabet
--------
| Token | Meaning |
| :-- | :-- |
| `.` or `b` | an off cell (value 0) |
| `o` | a full cell (value 255) |
| `A`..`X` | values 1..24 |
| `pA`..`yO` | values 25..255, in blocks of 24 |
| a leading number | repetition count of the symbol that follows |
| `$` | end of a row |
| `%`, `#`, `@A`, `@B`, … | end of each successive higher dimension |
| `!` | end of the pattern |

Values decode to `[0, 1]` floats by dividing by 255, which is where the format's
precision limit lives: a round trip is exact only to 1/255. `arr2rle` is
therefore lossy for continuous fields by construction, not by accident — §6's
round-trip test asserts `atol=1/255`, not equality.
"""

import itertools

import jax.numpy as jnp

__all__ = ["rle2arr", "arr2rle", "DIM_DELIM"]

Array = jnp.ndarray

#: Delimiter that closes each dimension. Index `i` closes the `i`-th axis
#: counting from the innermost, so a 2-D pattern uses `$` and a 3-D one adds `%`.
DIM_DELIM = {0: "", 1: "$", 2: "%", 3: "#", 4: "@A", 5: "@B", 6: "@C", 7: "@D", 8: "@E", 9: "@F"}

_LEAD = "pqrstuvwxy@"


def _token_value(token: str) -> int:
    """One RLE symbol -> its 0..255 cell value."""
    if token in ".b":
        return 0
    if token == "o":
        return 255
    if len(token) == 1:
        return ord(token) - ord("A") + 1
    return (ord(token[0]) - ord("p")) * 24 + (ord(token[1]) - ord("A") + 25)


def _value_token(value: int) -> str:
    """A 0..255 cell value -> its RLE symbol (two characters above 24)."""
    if value == 0:
        return "."
    if value < 25:
        return chr(ord("A") + value - 1)
    return chr(ord("p") + (value - 25) // 24) + chr(ord("A") + (value - 25) % 24)


def _push(stack: list, item, count: str, repeat: bool) -> None:
    """Append `item`, then `count - 1` more copies (or empties, for delimiters)."""
    stack.append(item)
    if count:
        stack.extend([item if repeat else []] * (int(count) - 1))


def _max_lengths(depth: int, level: int, nested: list, lengths: list[int]) -> None:
    lengths[level] = max(lengths[level], len(nested))
    if level < depth - 1:
        for child in nested:
            _max_lengths(depth, level + 1, child, lengths)


def _cubify(depth: int, level: int, nested: list, lengths: list[int]) -> None:
    """Pad a ragged nested list out to a rectangular block, with zeros."""
    missing = lengths[level] - len(nested)
    if level < depth - 1:
        nested.extend([[] for _ in range(missing)])
        for child in nested:
            _cubify(depth, level + 1, child, lengths)
    else:
        nested.extend([0] * missing)


def rle2arr(text: str) -> Array:
    """Decode an RLE string to a dense array.

    Args:
        text: the RLE string; a trailing `!` is optional. Its dimensionality is
              inferred from which delimiters appear, so no `d` argument is
              needed — `$` alone is 2-D, `$%` is 3-D, and so on.

    Returns:
        (zyx), values in `[0, 1]`. Note the axis order is the order the string
        is written in — outermost delimiter first — which is row-major, so the
        result indexes as `[z][y][x]`. Ragged rows are zero-padded to a
        rectangular block.
    """
    delimiters = list(DIM_DELIM.values())
    depth = 1
    for level, delimiter in DIM_DELIM.items():
        if delimiter and delimiter in text:
            depth = max(depth, level + 1)

    stacks: list[list] = [[] for _ in range(depth)]
    lead, count = "", ""
    for char in text.rstrip("!") + DIM_DELIM[depth - 1]:
        if char.isdigit():
            count += char
        elif char in _LEAD:
            lead = char  # a two-character value token, or the start of "@A"
        else:
            token = lead + char
            if token not in delimiters:
                _push(stacks[0], _token_value(token) / 255, count, repeat=True)
            else:
                # Closing dimension `i` folds every stack below it upward.
                level = delimiters.index(token)
                for below in range(level):
                    _push(stacks[below + 1], stacks[below], count, repeat=False)
                    stacks[below] = []
            lead, count = "", ""

    nested = stacks[depth - 1]
    lengths = [0] * depth
    _max_lengths(depth, 0, nested, lengths)
    _cubify(depth, 0, nested, lengths)
    return jnp.asarray(nested, dtype=jnp.float32)  # (zyx)


def _drill(depth: int, level: int, nested, row_fn):
    if level < depth - 1:
        return [_drill(depth, level + 1, child, row_fn) for child in nested]
    return row_fn(nested)


def _join(depth: int, level: int, nested, row_fn) -> str:
    delimiter = DIM_DELIM[depth - 1 - level]
    if level < depth - 1:
        return delimiter.join(_join(depth, level + 1, child, row_fn) for child in nested)
    return delimiter.join(row_fn(nested))


def arr2rle(array: Array, shorten: bool = True) -> str:
    """Encode a dense array as an RLE string.

    Args:
        array:   (zyx), values in `[0, 1]`. Quantized to 0..255 on the way out,
                 so the round trip is exact only to `1/255`.
        shorten: run-length-encode repeats; False writes one token per cell,
                 which is useful when diffing two patterns by eye.

    Returns:
        the RLE string, terminated by `!`.
    """
    array = jnp.asarray(array)
    depth = array.ndim
    values = jnp.rint(jnp.clip(array, 0.0, 1.0) * 255).astype(int).tolist()
    if shorten:
        runs = _drill(
            depth,
            0,
            values,
            lambda row: [(len(list(g)), _value_token(v)) for v, g in itertools.groupby(row)],
        )
        text = _join(depth, 0, runs, lambda row: [(str(n) if n > 1 else "") + c for n, c in row])
    else:
        text = _join(depth, 0, values, lambda row: [_value_token(v) for v in row])
    return text + "!"
