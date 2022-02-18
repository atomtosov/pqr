# pqr

pqr is a Python library for portfolio quantitative research.

Provides:

1. Library for testing factor strategies
2. A lot of different statistical metrics for portfolios
3. Fancy visualization of results

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pqr.

```bash
pip install pqr
```

## Documentation

You can find it on [rtd](https://pqr.readthedocs.io/en/latest/index.html) (for now documentation is outdated).

## Quickstart

```python
import pandas as pd
import pqr

# read and preprocess the data
prices = pd.read_csv("prices.csv", parse_dates=True)
pe = pd.read_csv("pe.csv", parse_dates=True)
volume = pd.read_csv("volume.csv", parse_dates=True)
prices, pe, volume = pqr.utils.replace_with_nan(prices, pe, volume, to_replace=0)

# define universe and make benchmark based on it
universe = pqr.Universe(prices)
universe.filter(volume >= 10_000_000)

benchmark = pqr.Benchmark.from_universe(
    universe,
    allocation_algorithm=pqr.AllocationAlgorithm([
        pqr.utils.EqualWeights(),
    ])
)

# prepare value factor
preprocessor = pqr.factors.Preprocessor([
    pqr.factors.Filter(universe.mask),
    pqr.factors.LookBackMedian(3),
    pqr.factors.Hold(3),
])

value = pqr.factors.Factor(
    pe,
    better="less",
    preprocessor=preprocessor
)

# form a factor model, covering all stock universe, and build portfolios
fm = pqr.factors.FactorModel(
    universe=universe,
    strategies=pqr.factors.split_quantiles(3),
    allocation_algorithm=pqr.AllocationAlgorithm([
        pqr.utils.EqualWeights()
    ]),
    add_wml=True
)

portfolios = fm.backtest(value)

# create a dashboard with basic info about our strategies
summary = pqr.dashboards.Dashboard([
    pqr.dashboards.Chart(pqr.metrics.CompoundedReturns(), benchmark=benchmark),
    pqr.dashboards.Table([
        pqr.metrics.MeanReturn(annualizer=1, statistics=True),
        pqr.metrics.Volatility(annualizer=1),
        pqr.metrics.SharpeRatio(rf=0),
        pqr.metrics.MeanExcessReturn(benchmark),
        pqr.metrics.Alpha(benchmark, statistics=True),
        pqr.metrics.Beta(benchmark),
    ])
])

summary.display(portfolios)
```

You can also see this example on real data with output in `examples/quickstart.ipynb`.

## Communication

If you find a bug or want to add some features, you are welcome to telegram @atomtosov or @eura71.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Project status

Now the project is in beta-version.
