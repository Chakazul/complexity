# Vendored original Lenia catalogue

Patterns from the original Lenia (`~/github/lenia-ref`, commit
`adfc542939266de7f4bb7ebb552e8499701ee107`), `code/Lenia/Python/`. Lenium does not ship
this catalogue yet; when it does, load the patterns from there instead (PLAN.md §4.2).

| File | Source |
|---|---|
| `animals.json` | `animals.json`: 26 of its 614 entries, verbatim, each with its original `index` added. Lenia creatures 11, 28, 164, 233, 89, 96, 498, 283, 348, 588 (PLAN.md §4.2), and the Game of Life patterns 598–613 |
| `rle.py` | `LeniaND.py`: `Board.ch2val` verbatim, `Board.rle2arr` reduced to 2-D |

The creatures were found with the exponential *bump* kernel core (`kn = 1`); they run here
on the Gaussian kernel shell instead, which a 1000–4000-step test showed they survive
(PLAN.md §4.2). The Life patterns (`kn = 4` in Lenia) are standard Life RLE.
