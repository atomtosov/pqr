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
prices, pe, volume = pqr.utils.align(prices, pe, volume)

# define universe and make benchmark based on it
universe = volume >= 1_000_000
returns_calculator = pqr.utils.partial(
    pqr.calculate_returns,
    universe_returns=pqr.prices_to_returns(prices),
)

benchmark = pqr.Benchmark.from_universe(
    universe=universe,
    allocator=pqr.equal_weights,
    calculator=returns_calculator,
)

# prepare dashboard
table = pqr.metrics.Table()
table.add_metric(
    pqr.utils.partial(
        pqr.metrics.mean_return,
        statistics=True,
        annualizer=1,
    ),
    multiplier=100,
    precision=2,
    name="Monthly Mean Return, %",
)
table.add_metric(
    pqr.utils.partial(
        pqr.metrics.volatility,
        annualizer=1,
    ),
    multiplier=100,
    precision=2,
    name="Monthly Volatility, %",
)
table.add_metric(
    pqr.metrics.max_drawdown,
    multiplier=100,
    name="Maximum Drawdown, %",
)
table.add_metric(
    pqr.utils.partial(
        pqr.metrics.mean_excess_return,
        benchmark=benchmark,
        statistics=True,
        annualizer=1,
    ),
    multiplier=100,
    precision=2,
    name="Monthly Mean Excess Return, %",
)
table.add_metric(
    pqr.utils.partial(
        pqr.metrics.alpha,
        benchmark=benchmark,
        statistics=True,
        annualizer=1,
    ),
    multiplier=100,
    precision=2,
    name="Monthly Alpha, %",

)
table.add_metric(
    pqr.utils.partial(
        pqr.metrics.beta,
        benchmark=benchmark,
        statistics=True,
    ),
    precision=2,
    name="Monthly Beta, %",
)

fig = pqr.metrics.Figure(
    pqr.metrics.compounded_returns,
    name="Compounded Returns",
    benchmark=benchmark,
    kwargs={
        "figsize": (10, 6),
    }
)

summary = pqr.metrics.Dashboard([table, fig])

# prepare value factor
static_transform = pqr.utils.compose(
    pqr.utils.partial(pqr.factors.filter, universe=universe),
    pqr.utils.partial(pqr.factors.look_back_median, period=3),
    pqr.utils.partial(pqr.factors.hold, period=3),
)

value = static_transform(pe)

# form a factor model, covering all stock universe, and build portfolios
portfolios = pqr.factors.backtest_factor_portfolios(
    factor=value,
    strategies=pqr.factors.split_quantiles(3, "less"),
    allocator=pqr.equal_weights,
    calculator=returns_calculator,
    add_wml=True,
)

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
