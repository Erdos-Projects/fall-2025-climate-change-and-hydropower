import pandas as pd

#single-dam-test-WA-622.csv is got from godeeep-hydro-historical.monthly.py The link to it is here:
# https://zenodo.org/records/14269763
#single-dam-test-precip is a txt to csv by from climdiv-pcpncy.

df1 = pd.read_csv("single-dam-test-precip-WA-45065.csv")
df2 = pd.read_csv("single-dam-test-WA-622.csv")

#We are concerned with getting percipitation data from 1982 - 2019. OF course we can change these
#As we use different data sets.
df1 = df1.loc[87:124,:]

#The percipitation data is written to have all 12 months of a year on one row, but the data
#Im working with in df2 has a month per row so I manually transpose each row on df1 to match
#df2 being one month per column.
df1_T = df1.iloc[0,1:13].T
for i in range(37):
    dfn_T = df1.iloc[1+i,1:13].T
    df1_T = pd.concat([df1_T,dfn_T],axis=0)

#We concatenate the two data frames such that a given month and years percip lines up with its energy output
df_complete = pd.concat([df2.reset_index(drop=True),df1_T.reset_index(drop=True)],axis=1)

#Name percipitation column
df_complete.rename(columns={0:"Precipitation"},inplace=True)


#Making sure our output looks good and outputs it to a csv file.
print(df_complete.head())
output_name = "ShowcaseSingleWADam.csv"
df_complete.to_csv(output_name,index=False)
