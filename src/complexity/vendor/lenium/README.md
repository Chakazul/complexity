# Vendored Lenium code and data

Lenium (the JAX Lenia library) is not published yet, so the parts this project needs are
copied here. The base is the **hand-coded Lenium**, `~/github/lenium.old` (identical to
`~/github/lenia-ref/code/lenium`), not the agent-written rewrite in `~/github/lenium`. When
Lenium is published, replace this directory with `import lenium`, add the dependency, and
keep the golden tests in `tests/` as the check that behaviour is unchanged (PLAN.md §4.2).

| File | Source (in `~/github/lenium.old`, commit `9acb864f82f1d4bea048a985ef3d3da864fb973c`) |
|---|---|
| `patterns.py` | `src/lenium/pattern.py`: entries `orbium`, `min-orbium` and `aquarium`, verbatim |
| `model.py` | Ported to JAX functions from `src/lenium/model.py` (`Lenia`, `IntegerLenia`), `func.py` (`bell`, `calc_kernels`, `calc_growth_lookup`, `discretize`, `get_params`) and `util.py` (`direct_convolve`); single channel and kernel, and multi-channel `Lenia` (2-D). Differences are listed in its docstring |

The multi-channel port has the same kernel FFT as Lenium (bit-identical) and differs by
2.4e-7 after one step on CPU (float rounding order), growing chaotically, as for
single-channel Lenia. Lenium's own runs are not reproducible on the GPU.
