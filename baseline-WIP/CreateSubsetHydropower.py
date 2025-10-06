import csv
import pandas as pd

input_file = "percipitation.csv"
output_file = "single-dam-test-precip-WA-45065.csv"

data = pd.read_csv(input_file)
data["ID"] = data["ID"].astype(str)
data[(data["ID"].str.startswith("45065"))].to_csv(output_file,index=False)