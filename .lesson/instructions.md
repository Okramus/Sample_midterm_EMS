# Sample Midterm Economic Modeling and Simulation

* Important: the midterm is different from the sample in significant ways, so if you just copy-paste the code your solutions will be most likely wrong and your grade will be penalized. If I detect errors that show that you have __copied from another classmate, your grade will me penalized__ even more.

* The maximum score is 100 points, and you can get an additional 15 pts if your code is __particulary clean and original__; in other words, you can score less than 100 point in the exercises and still get a 100.

* This exam is __open-book__: you may also lookup on the internet as long as you do not communicate with your classmates or anyone else.

* The total amount of points is 100, but you can get up to 15 bonus points if your code is clean and/or elegant. In other words, you do not need to get everything right to get the maximum grade. However, it is __critical that the code runs__ and that there are no execution errors: a code that runs but that misses some calculations will be graded benevolently; a code that does not run will not.

* The midterm involves quite a lot of __plotting__; I have used the library `matplotlib.pyplot` extensively in the sample midterm because I find it easier for you to apply to time series, compared to `seaborn`. However, you are of course free to use `seaborn` if you prefer.

* For the exercises below you will need the following libraries, parameters, and code to create the path for the plots:

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import statsmodels.formula.api as smf
import pmdarima as pm
```

```python
from pathlib import Path
```

```python
plt.style.use("seaborn-darkgrid")
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["text.color"] = "k"
matplotlib.rcParams["figure.dpi"] = 200
```

```python
directory = "plots"
Path(directory).mkdir(parents=True, exist_ok=True)
```

## Exercise 1: exploratory analysis and OLS [40 pts]

(a) [5 pts] Load the dataset `car_crashes` from `seaborn`. Print it and print the name of its columns. It contains the number of people involved in car crashes per billion miles, and the percentage of those people who were "speeding", under the effects of "alcohol", and so on. Each row represents a state in the US.

(b) [5 pts] Create a new column called `n_speeding` that contains the number of people (not the percentage) per billion miles who were speeding. 

(c) [5 pts] Plot a histogram with the distribution of the percentages for `speeding` and for `alcohol`.

(d) [5 pts] Define a `LeastSquares` function like the one we defined in class, that provides the intercept and slope after getting as arguments an `x` and `y` vectors.

(e) [5 pts] Apply the function to `alcohol` and `total`, in order to understand if there is a correlation between the percentage of drivers under the effects of `alcohol` and the total number of people involved in an accident.

(f) [5 pts] Build a column called `fit_alcohol` that contains the model fit for `total` for the datapoints in the dataset; do the same thing for `fit_speeding`.

(g) [10 pts] Draw a scatterplot that includes `total` in the `y` axis, and `alcohol` and `speeding` in the `x` axis; plot both the actual dataset and your fits; for the fits, you do not need to use lines, you can use squared markers for simplicity.

## Exercise 2: logistic regression with statsmodels [25 pts]

(a) [5 pts] Load the `planets` dataset from `seaborn`.

(b) [5 pts] Create a new column called `twentyfirst_century` that is equal to `1` when the planet has been discovered in the 21st century and `0` otherwise.

(c) [5 pts] With `statsmodels`, buils a logistic regression model that takes the orbital period, the mass, and the distance as regressors and predicts whether a planet was likely to be discovered in the 21st century (versus 20th century). Get the summary of the results of the model.

(d) [5 pts] Build a dataframe that contains data for a hypothetical new planet with orbital period of `100`, mass of `1`, and distance of `100`.

(e) [5 pts] Predict the chances that this planet has been discovered in the 21st century (versus in the 20th).

## Exercise 3: time series [35 pts]

(a) [5 pts] Read the file `AirQualityUCI.csv`, which is in the `lesson/assets` folder. Specify that the separator is a semicolon.

(b) [5 pts] Transform the `Date` column into date format with the `to_datetime` command from `pandas`.

(c) [5 pts] Sort the values of the dataframe by date, and apply a filter to get only the datapoints with dates after "2004-04-01" and before "2004-12-31".

(d) [5 pts] Create a copy of the dataframe and apply an exponential smoothing with span equal to 30 to the column `NOx(GT)`; you may overwrite the `NOx(GT)` column or create a new one.

(e) [5 pts] Plot the original `NOx(GT)` column against the exponentially smoothed one, in different colors.

(f) [5 pts] Plot the autocorrelation plot of the smoothed column.

(g) [5 pts] Use `autoarima` from `pmdarima` to find the optimal arima model for the smoothed `NOx(GT)`. Embed the code into a `StepwiseContext` loop to avoid overuse of the CPU. Print a summary of the results.