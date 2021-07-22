# pqr

pqr is a Python library for portfolio quantitative research.

Provides:
  1. Library for testing factor strategies
  2. A lot of different statistical metrics for portfolios
  3. Fancy visualization of results

## Installation

For now download code directly from github and copy it to directory, where you work, or add it to path.

But pqr will be soon on PyPI!

## Documentation

You can find it on [rtd](https://pqr.readthedocs.io/en/latest/index.html).

## Quickstart

```python
import matplotlib.pyplot as plt
import pandas as pd

import pqr

# read data
prices = pd.read_csv('prices.csv', parse_dates=True)
pe = pd.read_csv('pe.csv', parse_dates=True)
volume = pd.read_csv('volume.csv', parse_dates=True)

# preprocess the data
prices, pe, volume = pqr.correct_matrices(prices, pe, volume)
prices, pe, volume = pqr.replace_with_nan(prices, pe, volume,
                                          to_replace=[0, 'nan'])

# go to factors
value = pqr.factorize(
    factor=pe,
    is_dynamic=False,
    looking_period=3,
    lag_period=0,
    holding_period=3
)

liquidity = pqr.factorize(
    factor=volume,
    is_dynamic=False,
    looking_period=1,
    lag_period=0,
    holding_period=1
)
liquidity_threshold = pqr.Thresholds(lower=10_000_000)

# create custom benchmark from liquid stocks with equal weights
benchmark = pqr.benchmark_from_stock_universe(
    prices,
    filtering_factor=liquidity,
    filtering_thresholds=liquidity_threshold
)

# fitting the factor model on value factor (3-0-3)
# after fit we will get 3 quantile portfolios and wml-portfolio
portfolios = pqr.fit_factor_model(
    prices,
    value,
    filtering_factor=liquidity,
    filtering_thresholds=liquidity_threshold
)


# fetch the table with summary statistics and plot cumulative returns
summary = pqr.compare_portfolios(*portfolios, benchmark=benchmark)
print(summary)

# and show the plot of cumulative returns
pqr.plot_cumulative_returns(*portfolios, benchmark=benchmark)
plt.show()
```

You can also see this example on real data with output in examples/quickstart.ipynb.

## Communication
If you find a bug or want to add some features, you are welcome to telegram @atomtosov or @eura71.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Project status
Now the project is in beta-version.
