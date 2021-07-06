# pqr

pqr is a Python library for portfolio quantitative research. It helps to test factor strategies easily.

## Installation

For now download code directly from github and copy it to directory, where you work, or add it to path.

But pqr will be soon on PyPI!

## Quickstart

```python
import matplotlib.pyplot as plt
import pandas as pd

from pqr.preprocessing import correct_matrices, replace_with_nan
from pqr.factors import PickingFactor, FilteringFactor
from pqr.factor_model import FactorModel

# read data
prices = pd.read_csv('prices.csv', parse_dates=True)
pe = pd.read_csv('pe.csv', parse_dates=True)
volume = pd.read_csv('volume.csv', parse_dates=True)

# preprocess the data
prices, pe, volume = correct_matrices(prices, pe, volume)
prices, pe, volume = replace_with_nan(prices, pe, volume, to_replace=[0, 'nan'])

# go to factors
value = PickingFactor(
    data=pe,
    dynamic=False,
    bigger_better=False,
    name='value'
)
liquidity = FilteringFactor(
    data=volume,
    dynamic=False,
    min_threshold=10_000_000,
    name='liquidity'
)

# creating factor model
fm = FactorModel(
    looking_period=3,
    lag_period=0,
    holding_period=3
)

# fitting the factor model on value factor
# after fit we will get 3 quantile portfolios and wml-portfolio
fm.fit(
    prices=prices,
    picking_factor=value,
    filtering_factor=liquidity,
    n_quantiles=3,
    add_wml=True
)

# fetch the table with summary statistics and plot cumulative returns
summary = fm.compare_portfolios(plot=True)

# then we can print stats
print(summary)

# and show the plot
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
