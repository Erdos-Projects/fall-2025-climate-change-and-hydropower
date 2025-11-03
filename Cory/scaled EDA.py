import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#-------------------------
# 1. Deseasonalize data
#-------------------------
def deseasonalize(df, groupby, cols):
    """
    Inserts new columns that are the averages / SDev of columns across groups

    Args:
        df - the dataframe to be used
        groupby - which columns to groupby. Should be a list of column names
        cols - which columns to summarize over. Must be a list

    Ex:
    >>> deseasonalize(df, ['month', 'state'], ['pcpn', 'tmin'])
    This will create new columns for the average pcpn and tmin, grouped by month and state
    """
    grouped = df.groupby(groupby)
    for feature_name in cols:
        df['avg '+feature_name] = grouped[feature_name].transform('mean')
        df['std '+feature_name] = grouped[feature_name].transform('std')
        df['z '+feature_name] = (df[feature_name] - df['avg '+feature_name]) / df['std '+feature_name]


#-----------------------------------------
# 2. Visualize seasonal timeseries data
#-----------------------------------------
def plot_trend(ax, df, groupby, feature_name, quantiles = [0.01, 0.1, 0.25, 0.75, 0.9, 0.99], alpha = [.1, .4, .8, .4, .1]):
    """
    Plots quantile data of a dataframe's columns against time

    Args:
        ax - pyplot axis
        df - dataframe
        groupby - the column to group by
        feature_name - the column to show quantile data of
        quantiles - which cutoffs to use
        alpha - transparency: alpha[i] is the transparency of []
    """
    results = []
    x = []
    for label, group in df.groupby(groupby):
        results.append(group[feature_name].quantile(quantiles))
        x.append(label)
    x = np.asarray(x)
    results = np.asarray(results)
    for i in range(len(quantiles)-1):
        ax.fill_between(x, results[:,i], results[:, i+1], color = 'red', alpha = alpha[i], linewidth=0,
                         label = f'{quantiles[i]}-{quantiles[i+1]}')
    ax.set_xlabel(groupby)
    ax.set_ylabel(feature_name)

def seasonal_and_trend_plot(df, feature_names):
    fig, axs = plt.subplots(len(feature_names), 2)
    for i, feature in enumerate(feature_names):
        plot_trend(axs[i, 0], df, 'year', feature)
        plot_trend(axs[i, 1], df, 'month', feature)

#---------------------------------
# 3. Pairplotting
#---------------------------------
def pairplot(df):
    original = df[['tmin', 'tavg', 'tmax', 'pcpn', 'RectifHyd_MWh']]
    sns.pairplot(original, plot_kws = {'alpha':0.1, 's':1})
    plt.show()
    zs = df[['z tmin', 'z tavg', 'z tmax', 'z pcpn', 'z RectifHyd_MWh']]
    sns.pairplot(zs, plot_kws = {'alpha':0.1, 's':1})
    plt.show()


# I wanted to try basic Linear regresion on the Zs, I got an R^2 of about 0.06, so I'll scrap that for now
if __name__ == '__main__':
    df = pd.read_csv('train_val_set.csv')
    print('Deseasonalizing')
    deseasonalize(df, ['Latitude', 'Longitude', 'month'], ['RectifHyd_MWh', 'tmin', 'tmax', 'tavg', 'pcpn'])
    seasonal_and_trend_plot(df, ['tmin', 'tavg', 'tmax', 'pcpn', 'RectifHyd_MWh'])
    plt.show()
    pairplot(df)

