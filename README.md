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

You can find it on [rtd](https://pqr.readthedocs.io/en/latest/index.html).

## Quickstart

```python
import pandas as pd
import numpy as np

import pqr

# read data
prices = pd.read_csv('prices.csv', parse_dates=True)
pe = pd.read_csv('pe.csv', parse_dates=True)
volume = pd.read_csv('volume.csv', parse_dates=True)

# preprocess the data
prices = prices.replace(0, np.nan)
pe = pe.replace(0, np.nan)
volume = volume.replace(0, np.nan)

# go to factors
value = pqr.Factor(pe)
value.look_back(3, 'static').lag(0).hold(3)

liquidity = pqr.Factor(volume).look_back()
liquidity_filter = liquidity.data >= 10_000_000

value.prefilter(liquidity_filter)

# create custom benchmark from liquid stocks with equal weights
benchmark = pqr.Benchmark().from_stock_universe(prices,liquidity_filter)

# fitting the factor model on value factor (3-0-3)
# after fit we will get 3 quantile portfolios
portfolios = pqr.fit_factor_model(prices, value)
# fetch the table with summary statistics and plot cumulative returns
pqr.factor_model_tear_sheet(portfolios, benchmark)
```

You can also see this example on real data with output in
examples/quickstart.ipynb.

## Communication

If you find a bug or want to add some features, you are welcome to telegram
@atomtosov or @eura71.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Project status

Now the project is in beta-version.
