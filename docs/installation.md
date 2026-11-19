# Installation

SwingPy needs Python 3.10 or later, NumPy and SciPy.

```bash
pip install "swingpy @ git+https://github.com/mohammmadyazdani/swingpy.git"
```

Optional features are installed as extras:

| Extra | Adds | Use |
|---|---|---|
| `fast` | numba | compiled right-hand side, several times faster cycles |
| `plot` | matplotlib | figures |
| `test` | pytest | the test suite |

```bash
pip install "swingpy[fast,plot] @ git+https://github.com/mohammmadyazdani/swingpy.git"
```

From a clone of the repository, for development:

```bash
pip install -e ".[fast,plot,test,docs]"
pytest
mkdocs serve        # this documentation, locally
```
