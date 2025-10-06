import pandas as pd

df1 = pd.read_csv("single-dam-test-precip-WA-45065.csv")
df2 = pd.read_csv("single-dam-test-WA-622.csv")

df1 = df1.loc[87:124,:]

#for i in range(38):
#    for k in range(12):
#        precip_month = df1.iloc[i, 1 + k]
#        df2[(12 * i) + k, "Precip"] = precip_month

df1_T = df1.iloc[0,1:13].T
for i in range(37):
    dfn_T = df1.iloc[1+i,1:13].T
    df1_T = pd.concat([df1_T,dfn_T],axis=0)


df_complete = pd.concat([df2.reset_index(drop=True),df1_T.reset_index(drop=True)],axis=1)

df_complete.rename(columns={0:"Precipitation"},inplace=True)

#print(df1_T.shape[0])
#print(df2.shape[0])

#df = pd.concat([T1.reset_index(drop=True),T2.reset_index(drop=True)], axis=1)

print(df_complete.head())
output_name = "ShowcaseSingleWADam.csv"
df_complete.to_csv(output_name,index=False)