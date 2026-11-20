# NuQuLib

[![codecov](https://codecov.io/github/SotaYoshida/nuqulib/graph/badge.svg?token=2DAZLO1N3U)](https://codecov.io/github/SotaYoshida/nuqulib)

Nuclear Quantum Computing Library

For three-nucleon calculations, use `me3j.gz` input, as demonstrated in `tutorial_NuQuLib.ipynb`. This path includes angular-momentum and isospin recoupling in the direct JT-to-M conversion. The `readable.txt` reader is a legacy development path with known mixed proton-neutron conversion issues and is not recommended for production calculations.

## Hamiltonian counts and resource estimates

Edit the model-space settings at the top of `misc/counting_Hamiltonian_terms.jl`, then run from this checkout:

```sh
julia --threads=2 misc/counting_Hamiltonian_terms.jl
```

The script requires Julia 1.12 or later, `TimerOutputs`, and `StaticArrays`. It counts all configured spaces in one process and replaces `newTgateCounts.tsv` only after the complete batch passes validation. Counts include the identity and describe symmetry-allowed Jordan–Wigner operator patterns; they do not include interaction-specific coefficient cancellations. The original six TSV columns are retained, followed by cutoff and provenance metadata.

In the paper revision workspace, the figure notebook and its input TSV are kept together in the parent directory of this checkout. Run `drawing_Tcounts_trotter_error_NuQuLib_updated.ipynb` there to reproduce the resource plots and tables. Its default run uses the archived structural counts after verifying the Julia source hash. Set `REGENERATE_COUNTS = True` in the notebook to rebuild that TSV.

Run the counting and TSV regression checks with:

```sh
julia --threads=2 misc/test_counting_Hamiltonian_terms.jl
```

## Documentation

Detailed information on how to use NuQuLib, please refer to the official documentation:
https://sotayoshida.github.io/nuqulib/

## Authors

Primary developers: Sota Yoshida, Alessandro Baroni, Takayuki Miyagi, Ermal Rrapaj

## How to cite

If you use NuQuLib in your research, please cite the following reference: [arXiv:2607.08047](https://arxiv.org/abs/2607.08047)

```
@misc{arXiv:2607.08047,
      title={Nuclear Many-Body Systems as Benchmarks for Quantum Computing}, 
      author={Sota Yoshida and Alessandro Baroni and Takayuki Miyagi and Ermal Rrapaj},
      year={2026},
      eprint={2607.08047},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2607.08047}, 
}
```
