import csv
import pandas as pd

#Creating an arbitrary csv file to put the percipitation data for a single dams data into.
#I chose by looking for the closest fip code to the dam site manually. 
#Percipitation.csv is simply the percipitation data from climdiv-pcpncy-v1.0.0-20250905.txt converted to csv
input_file = "percipitation.csv"
output_file = "single-dam-test-precip-WA-45065.csv"

#The ID needs to read as a str so we keep leading zeros (they are important)
#We take ID's that start with 45065, corresponds to the state WA (45) and fip 065
data = pd.read_csv(input_file)
data["ID"] = data["ID"].astype(str)
data[(data["ID"].str.startswith("45065"))].to_csv(output_file,index=False)
