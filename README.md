# pyJASPER

Python implementation of Joint Bayesian Analysis of SPatial Expression via Regression (JASPER): a fully Bayesian module for joint detection of Spatially Varying Genes (SVGs) in spatial transcriptomics datasets.

JASPER uses a Negative-Binomial count regression model with a latent multivariate Gaussian expression surface modeled using spatial basis functions, a low-rank covariance structure, and SVG inticators.

The main Python3 implementation is in [`pyJASPER.py`](pyJASPER.py) which contains the `JASPER` class, while [`main.ipynb`](main.ipynb) contains an example run on a toy dataset generated therein.

## Requirements

- NumPy
- SciPy
- polyagamma

Install the Python dependencies with:

```bash
pip install numpy scipy polyagamma
```

## Usage

```python
from pyJASPER import JASPER

model = JASPER(
    C=C,          # n x p count matrix
    X=X,          # n x K spatial basis/covariate matrix
    N=N,          # n-vector of library sizes
    r=5,
    n_iter=2000,
    burn=1000,
    thin=1,
    seed=123,
)

model.run(verbose=True, show_every=50)

ppi = model.posterior_inclusion_probabilities()
selected = model.selected_genes_by_threshold(threshold=0.5)
selected_pefdr, threshold = model.selected_genes_by_pefdr(target=0.05)
```

## Outputs

After `run()`, posterior summaries are available through:

- `posterior_inclusion_probabilities()`
- `posterior_phi_mean()`
- `posterior_psi_mean()`
- `posterior_g_mean()`
- `selected_genes_by_threshold()`
- `selected_genes_by_pefdr()`

## Files

- [`pyJASPER.py`](pyJASPER.py): JASPER sampler implementation.
- [`main.ipynb`](main.ipynb): notebook for exploratory use.
