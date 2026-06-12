# Accelerated Fourier SAT (AFSAT): Fast (GPU Accelerated) Fourier-SAT Solver

AFSAT is a native pseudo-Boolean (PB) Continuous Local Search (CLS) Satisfiability (SAT) solver that leverages recent developments in closed-form transformations of various pseudo-Boolean constraint types into continuous multilinear polynomial on a one-to-one basis.

The approach takes advantage of the JAX ecosystem, a modern high performance scientific computing framework, leveraging automatic differentiation and efficient optimization for fast distributed GPU kernels.

This work is based on recent theoretical advances by Kyrillidis *et al*<sup>[<a href="#ref-1">1</a>]</sup> and a proof-of-concept demonstration by Cen *et al*<sup>[<a href="#ref-2">2</a>]</sup>. The code presented here is a new implementation from scratch, and supports arbitrary PB-SAT problems that can be expressed in any combination of the constraint types enumerated below.

A note on naming convention: Neither this code, nor the proof-of-concept work, uses a *fast Fourier* algorithm. The underlying transform is a *Fourier* transform, hence *Fourier SAT*.
After this transform, this solver utilises a second discrete Fourier transform for fast polynomial evaluation on GPU, from which the *fast* is derived. 
We engineer this idea into a fully featured solver for GPU accelerators, hence *Accelerated*.

## Installation

### Prerequisites

- Python >=3.10+ (for JAX>=0.7.0) (see https://docs.jax.dev/en/latest/installation.html)
- For JAX acceleration, a compatible GPU or TPU.
  
Optionally, see https://github.com/NVIDIA/JAX-Toolbox for Docker images that may work.

### Setup

1. Clone the repository:
```bash
    git clone https://github.com/cjchristopher/accelerated-fourier-sat.git
    cd accelerated-fourier-sat
```

2. Create and activate a virtual environment (recommended, your choice of environment management, we use venv here for illustration):
```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install JAX and dependencies
```bash
    pip install jax[cudaXX] tqdm sparklines jaxopt
```
   N.B. CUDA 12 is no longer receiving feature updates and future versions of JAX will not support it. 
   Nvidia deprecated support for many older architectures with the release of CUDA 13, and so an older version of JAX/CUDA may be required for your architecture.
   See [JAX Installation](https://docs.jax.dev/en/latest/installation.html#installation) for additional instructions.
   We leave the details of the JAX/CUDA installation to the user, and cannot guarantee compatability,
   e.g. if you have different versions of CUDA, or non Nvidia GPUs (note, only tested on CPU and Nvidia GPUs/CUDA).

## Input Format

The solver accepts files in standard DIMACS CNF format as well as an extended hybrid format that supports various constraint types.

### Standard DIMACS CNF Format

- Comment lines starting with 'c' or '*'
- A problem line starting with 'p cnf' followed by the number of variables and clauses
- Clauses represented as space-separated integers ending with '0'
- Positive integers represent positive literals, negative integers represent negative literals
- Example: `1 -2 3 0` represents (x₁ ∨ ¬x₂ ∨ x₃)

### Hybrid DIMACS Format

This solver extends the DIMACS format to support various constraint types using the following syntax:

1. Constraint lines either:
   - Start with `h` (hybrid marker), e.g., `h 1 2 0`
   - Start with constraint type directly, e.g., `xor 1 2 0`
   - Or both for explicitly clarity e.g., `h eo 1 2 0`
   - For standard CNF clauses, no prefix is needed, e.g., `1 2 0`, but prefix with a single `h` is supported.

2. Supported constraint types:

   - **CNF clauses**: Standard disjunction of literals

     `1 -2 3 0` or `h 1 -2 3 0`

   - **XOR constraints**: An odd number of literals must be true

     `xor 1 2 -3 0` or `h x 1 2 -3 0` or `x 1 2 -3 0`

   - **NAE (Not-All-Equal) constraints**: Not all literals can have the same value

     `nae 1 2 3 0` or `h n 1 2 3 0` or `n 1 2 3 0`

   - **EO (Exactly-One) constraints**: Exactly one literal must be true

     `eo 1 2 3 0` or `h e 1 2 3 0` or `e 1 2 3 0`

   - **AMO (At-Most-One) constraints**: At most one literal can be true

     `amo 1 2 3 0` or `h a 1 2 3 0` or `a 1 2 3 0`

   - **Ek (Exactly-k) constraints**: Exactly k literals must be true

     `ek <k> <literals> 0` or `h k <k> <literals> 0` or `k <k> <literals> 0`

     Where `<k>` is a non-zero positive integer:
       - `ek 2 1 2 3 0` (at least 2 variables must be true)
       - `ek 4 1 2 3 4 5 6 0` (more than 2 variables must be true)

   - **Cardinality constraints**: Specify how many literals in a set must be true

     `card <k> <literals> 0` or `h d <k> <literals> 0` or `d <k> <literals> 0`

     Where `<k>` can be:
     - A simple integer: `card 2 1 2 3 0` (at least 2 of the variables must be true)
     - A negative integer: `card -2 1 2 3 0` (fewer than 2 variables must be true)
     - With inequality prefix:
       - `card >=2 1 2 3 0` (at least 2 variables must be true)
       - `card >2 1 2 3 0` (more than 2 variables must be true)
       - `card <2 1 2 3 0` (fewer than 2 variables must be true)
       - `card <=2 1 2 3 0` (at most 2 variables must be true)

The parser is flexible with shorthand notations for constraint types, allowing for both single-character identifiers (`x`, `n`, `e`, `a`, `d`, `k`) and full names (`xor`, `nae`, `eo`, `amo`, `card`, `ek`).

## Usage

Run the solver with:

```bash
python afsat.py [options] input_file.cnf
```

### Command Line Options

- `-t, --timeout INT`: Maximum runtime in seconds (default: 300)
- `-b, --batch INT`: Batch size per GPU (default: -1 = compute heuristic maximum)
- `-r, --restart_thresh INT`: Number of batches before reweighting (default: 0, never reweight)
- `-f, --fuzz INT`: Number of times to attempt fuzzing per batch (default: 0)
- `-n, --n_devices INT`: Number of devices (e.g., GPUs) to use (default: all available, 0 = use all)
- `-i, --iters_desc INT`: Descent iteration depth (default: 100)
- `-d, --debug LEVEL`: Set logging level: DEBUG, INFO, WARNING, or ERROR (default: ERROR)
- `-e, --benchmark`: Enable benchmark mode (reduces output verbosity)
- `-c, --counting`: Counting mode - count/enumerate solutions until timeout (default: False (off))
- `-w, --warmup`: Perform a warmup run before starting the timer (may improve performance with JAX compilation cache)
- `-s, --rand_seed`: Randomize the random seed (default: uses fixed seed for reproducibility)
- `-p, --prefix FILE`: Path to file containing fixed variable assignments (one assignment per line, using SAT solver output format with negated literals)
- `-y, --profile`: Enable profiling (saves JAX trace to `/tmp/jax-trace` and device memory profile to `memory.prof`)
- `-u, --unsat_thresh FLOAT`: Implemented to replicate a benchmark in `FastFourierSAT` - treats a problem as solved when at least `(1-unsat-thresh)*#clauses` clauses are `True`
- `-m, --sample_meth STR`: New candidate assignments can be randomly sampled in various ways. Default is `bias`, options are `bias, coin, uniform, trunc`. Documentation TBD.
- `-q, --solver_tol FLOAT`: For convergence criteria solvers (e.g. gradient descent), overrides the default threshold for which convergence is deemed to have been met.
- `--stdout_log`: Sends output from logger (e.g. when `--debug` is set) to stdout instead of stderr.
- `--anomaly_quit`: Will cause the solver to immediately halt if numerical instability is detected (e.g. total evaluation is outside expected bounds)

### Examples

Basic usage:
```bash
    python afsat.py problem_file
```

Running with a 10-minute timeout and specific batch size:
```bash
    python afsat.py problem_file -t 600 -b 32
```

Enable five fuzzing passes per batch:
```bash
    python afsat.py problem_file -f 5
```

Run with debugging enabled and profiling:
```bash
    python afsat.py problem_file -d DEBUG -p
```

Use a prefix file with fixed variable assignments:
```bash
    python afsat.py problem_file -p prefix.txt
```

Use specific number of GPUs with custom iteration depth:
```bash
    python afsat.py problem_file_ -n 2 -i 200
```

Counting mode - find all solutions within 5 minutes:
```bash
    python afsat.py problem_file_ -t 300 -c 1
```

## Output

The solver outputs:
- Progress information during the search
- SAT/UNSAT result
- Variable assignments for satisfiable instances in DIMACS format (lines starting with "v")
- Timing and performance statistics

## Technical Details

The solver uses JAX for hardware acceleration and automatic differentiation. It transforms boolean satisfiability constraints into continuous functions that can be minimized using gradient-based methods.

### Solver Algorithms

AFSAT supports multiple optimization backends (default: `pgd`):

- **pgd**: Projected Gradient Descent with box constraints (JAXOPT)
- **lbfgsb**: Limited-memory BFGS with bounds (JAXOPT)
- **josp-lbfgsb**: JAXOPT's Scipy-wrapped L-BFGS-B
- **sp-lbfgsb**: Scipy L-BFGS-B with JAX compiled objective function

Note: The solver algorithm is currently hardcoded in the source. Support for command-line selection may be added in future versions.

## License

Dual-licensed under:

- Apache License 2.0 (`Apache-2.0`)
- GNU General Public License v2.0 or later (`GPL-2.0-or-later`)

You may choose either license when using, modifying, or redistributing this project.

Source files use the combined SPDX expression:

```
SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
```

License texts: see `LICENSE-APACHE` (Apache 2.0) and `LICENSE` (GPL v2).

Contributions are accepted under the same dual-license terms.

### Citation
If you use AFSAT in your research, please cite both the tool paper and the software artifact.
The two companion papers below are currently submitted for publication; fields such as DOI, pages, and publisher are placeholders pending acceptance and final publication metadata.
Until venue metadata is finalized, cite the manuscript (or arXiv preprint, when available) together with the software artifact.
For software citation, cite the exact version used, preferring a release DOI, then a release tag, otherwise an exact commit hash.

```bibtex
@misc{christopher2026afsat,
  title={Accelerated Fourier SAT (AFSAT): Fully Realising a GPU-based Symmetric Pseudo-Boolean SAT Solver}, 
  author={Cody J Christopher and Charles Gretton},
  year={2026},
  eprint={2606.06641},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2606.06641}, 
}

@misc{christopher2026clsstudy,
  title={A Study of Parallel Continuous Local Search}, 
  author={Cody J Christopher and Charles Gretton},
  year={2026},
  eprint={2606.06656},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2606.06656}, 
}

@software{christopher26afsat-git,
  title={Accelerated Fourier SAT (AFSAT): Fast (GPU Accelerated) Fourier-SAT Solver},
  author={Christopher, Cody and Gretton, Charles},
  year={2026},
  url={https://github.com/cjchristopher/accelerated-fourier-sat}
}
```

#### References
<a id="ref-1"></a>
1: Kyrillidis, A., Shrivastava, A., Vardi, M. Y., & Zhang, Z. (2021).
*Solving hybrid Boolean constraints in continuous space via multilinear Fourier expansions*.
Artificial Intelligence, 299, 103559.
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.artint.2021.103559-blue?logo=doi)](https://doi.org/10.1016/j.artint.2021.103559)
[![arXiv](https://img.shields.io/badge/PDF-Paper-red)](https://akyrillidis.github.io/pubs/Journals/fourierSAT.pdf)
[![GitHub](https://img.shields.io/badge/GitHub-FourierSAT-181717?logo=github)](https://github.com/vardigroup/FourierSAT)

<a id="ref-2"></a>
2: Cen, Y., Zhang, Z., & Fong, X. (2025).
*Massively Parallel Continuous Local Search for Hybrid SAT Solving on GPUs*.
Proceedings of the AAAI Conference on Artificial Intelligence, 39(11), 11140-11149.
[![DOI](https://img.shields.io/badge/DOI-10.1609%2Faaai.v39i11.33211-blue?logo=doi)](https://doi.org/10.1609/aaai.v39i11.33211)
[![arXiv](https://img.shields.io/badge/cs.AI-arXiv:2308.15020-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2308.15020)
[![GitHub](https://img.shields.io/badge/GitHub-FastFourierSAT-181717?logo=github)](https://github.com/seeder-research/FastFourierSAT)
