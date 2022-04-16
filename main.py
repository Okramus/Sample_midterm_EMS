# %%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import statsmodels.formula.api as smf
import pmdarima as pm

# %%
from pathlib import Path

# %%
plt.style.use("seaborn-darkgrid")
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["text.color"] = "k"
matplotlib.rcParams["figure.dpi"] = 200

# %%
directory = "plots"
Path(directory).mkdir(parents=True, exist_ok=True)

# %% Ex 1
sns.get_dataset_names()
df = sns.load_dataset(
    "car_crashes"
)  # total per billions miles, next four columns are %
print(df)

# %%
df.columns

# %%
df["n_speeding"] = df["total"] * df["speeding"] / 100
# %%
display(df)
# %%
df.speeding.describe()
# %%
plt.hist(
    x=[df.speeding, df.alcohol],
    bins=20,
    label=["speeding", "alcohol"],
    color=["red", "blue"],
    rwidth=1,
)
plt.legend()
plt.savefig(directory + "/hist_speeding.png")

# %%
def LeastSquares(xs, ys):
    mean_x = np.mean(xs)
    var_x = np.var(xs)
    mean_y = np.mean(ys)
    cov = np.dot(xs - mean_x, ys - mean_y) / len(xs)
    slope = cov / var_x
    inter = mean_y - slope * mean_x
    return inter, slope


# %%
inter, slope = LeastSquares(df.speeding, df.total)
print(f"Alcohol: {inter=}; {slope=}")
df["fit_speeding"] = inter + slope * df["speeding"]

# %%
inter, slope = LeastSquares(df.alcohol, df.total)
print(f"Speeding: {inter=}; {slope=}")
df["fit_alcohol"] = inter + slope * df["alcohol"]

# %%
plt.scatter(x="alcohol", y="total", data=df, color="blue", label="alcohol", alpha=0.5)
plt.scatter(x="speeding", y="total", data=df, color="red", label="speeding", alpha=0.5)
plt.scatter(x="alcohol", y="fit_alcohol", data=df, color="cyan", marker="s", alpha=0.5)
plt.scatter(
    x="speeding", y="fit_speeding", data=df, color="orange", marker="s", alpha=0.5
)
plt.xlabel("pct")
plt.ylabel("ppl involved per bn mi")
plt.legend()
plt.savefig(f'{directory}/scatter.png')
# %% Ex 2

planets = sns.load_dataset("planets")
# %%
planets["twentyfirst_century"] = (planets.year > 2000) * 1
# %%
model = smf.logit(
    "twentyfirst_century ~ orbital_period + mass + distance", data=planets
)
results = model.fit()
results.summary()
# %%
new_planet = pd.DataFrame(
    data={"orbital_period": [100], "mass": [1], "distance": [100]}
)
# %%
y = results.predict(new_planet)
# %%
print(
    f"the chances of discovering such a planet in the 21st c. (vs in the 20th) are {y[0]}"
)

# %% Ex 3
air = pd.read_csv(".lesson/assets/AirQualityUCI.csv", sep=";")
air["Date"] = pd.to_datetime(air["Date"])
air = air.sort_values("Date").query('Date > "2004-04-01"').query('Date < "2004-12-31"')
smooth_air = air.copy()
smooth_air["NOx(GT)"] = air["NOx(GT)"].ewm(span=30).mean()
# %%
plt.plot(air["Date"], air["NOx(GT)"], label="nox", color="gray")
plt.plot(smooth_air["Date"], smooth_air["NOx(GT)"], label="nox ewma", color="darkred")
plt.legend()
plt.savefig("plots/ewma.png")
# %%
pd.plotting.autocorrelation_plot(smooth_air["NOx(GT)"])
# %%
with pm.StepwiseContext(max_dur=15):
    model = pm.auto_arima(smooth_air["NOx(GT)"], stepwise=True, error_action="ignore")
results = model.fit(smooth_air["NOx(GT)"])
print(results.summary())
# %%
