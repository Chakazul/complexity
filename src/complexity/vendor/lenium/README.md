# Vendored Lenium code and data

Lenium (the JAX Lenia library) is not published yet, so the parts this project needs are
copied here. When Lenium is published, replace this directory with `import lenium`, add
the dependency, and keep the golden tests in `tests/` as the check that behaviour is
unchanged (PLAN.md §4.2).

| File | Source | Commit |
|---|---|---|
| `rle.py` | Lenium `src/lenium/io/rle.py`, verbatim | `139492ce0227ec2556f6455901f681676cfaace7` |
| `profiles.py` | Lenium `src/lenium/core/profiles.py`, verbatim | `139492ce0227ec2556f6455901f681676cfaace7` |
| `creatures.json` | Lenium `src/lenium/io/data/default_creatures.json`, entries `orbium` and `min_orbium` (number lists reformatted onto one line) | `139492ce0227ec2556f6455901f681676cfaace7` |
| `integer.py` | Ported to JAX from the predecessor library `~/github/lenia-ref/code/lenium/src/lenium/model.py` (`IntegerLenia`) and `func.py` (`calc_kernels`, `calc_growth_lookup`, `discretize`), `util.py` (`direct_convolve`); Lenium has no Integer Lenia engine yet | predecessor is not under git |
