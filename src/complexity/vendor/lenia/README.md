# Vendored original Lenia catalogue

Patterns from the original Lenia (`~/github/lenia-ref`, commit
`adfc542939266de7f4bb7ebb552e8499701ee107`), `code/Lenia/Python/`. Lenium does not ship
this catalogue yet; when it does, load the patterns from there instead (PLAN.md §4.2).

| File | Source |
|---|---|
| `animals.json` | `animals.json`: 30 of its 614 entries, verbatim, each with its original `index` added. Lenia creatures 11, 44, 173, 233, 91, 96, 498, 313, 355, 404, 408, 587, 257, 267 (PLAN.md §4.2), and the Game of Life patterns 598–613 |
| `rle.py` | `LeniaND.py`: `Board.ch2val` verbatim, `Board.rle2arr` reduced to 2-D |

The creatures were found with the exponential *bump* kernel core (`kn = 1`,
`exp(4 − 1/(r(1 − r)))`) and run on it here; patterns without `kn` (Lenium's) use the
Gaussian shell. All use Gaussian growth (`gn = 1`). The Life patterns (`kn = 4` in Lenia)
are standard Life RLE.
