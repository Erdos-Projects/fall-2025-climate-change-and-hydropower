
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


#Lets create a risk analysis for just one plant and provide it as a proof of concept.

#Create a baseline linear regression.
df = pd.read_csv("ShowcaseSingleWADam.csv")

x = df[["Precipitation"]].values
y = df[["power_predicted_mwh"]].values


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

#In texting I need y_test and y_train to be 1d numpy lists. this basically goes from pd -> np
y_test = y_test.ravel()
y_train = y_train.ravel()

#For quantile regression (examining the range of outputs for the median,10th and 90th percentile)
quantiles = [0.1,0.5,0.9]
models = {}

#Simple model. Again proof of concept, I plan to actually go through stronger models and have something
#of note by wednesday/thursday.
for q in quantiles:
    model = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    models[q] = model

# Prediction for quantiles
preds = {q: models[q].predict(X_test) for q in quantiles}

# -----------------------------
# Visualization (THIS STUFF IS CHATGPT!!!! Data visualization will be a task that we will have to do
#But for now, something simple so we have something to look at.)
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(y_test, label="Observed", color="black")

plt.plot(preds[0.5], label="Median prediction", color="blue")
plt.fill_between(range(len(y_test)), preds[0.1], preds[0.9],
                 color="blue", alpha=0.2, label="10th-90th percentile")

plt.legend()
plt.title("Probabilistic Forecast")
plt.xlabel("Time index")
plt.ylabel("Generation")
plt.savefig("test.jpg", dpi=300, bbox_inches="tight")
plt.close()
