This summarises all the changes I have included.

Found a dataset that includes details about 2534 dams in the US that are capable of generating hydropower. The dataset includes details such as:

Dam Height (Ft),
Hydraulic Height (Ft),
Structural Height (Ft),
NID Height (Ft),
NID Height Category,
Dam Length (Ft),
Volume (Cubic Yards),
NID Storage (Acre-Ft),
Max Storage (Acre-Ft),
Normal Storage (Acre-Ft),
Surface Area (Acres),
Drainage Area (Sq Miles),
Max Discharge (Cubic Ft/Second),
Spillway Type Spillway Width (Ft),
Latitude,
Longitude,
State,
County,
City,
Distance to Nearest City (Miles),
River or Stream Name and some other not so important details. 

Here is a link to the dataset from NID that I have put in my Google Drive for easy access (in CSV format): [link](https://drive.google.com/file/d/1SSTUEoitEWkfp5d7WKd57qBjCSg3i13Y/view?usp=sharing)

Here is a link to the source: [link](https://nid.sec.usace.army.mil/nid/#/dams/search/sy=@purposeIds:(6)&viewType=map&resultsType=dams&advanced=false&hideList=false&eventSystem=false)

1. I have added a new python script based on Cory's script for extracting data from USGS. I have made it such that it can extract data from 30 years ago till now (though for the test lat and long some of these doesn't seem to be available). The code can also extract multiple input features at the same time, it looks for the closest station that measures the parameter with a limit of 100 km and then extract the mean and if that is not available the instantaneous value for the parameter. It also combines all of these to one table.

2. I have added a folder named lightGBM which uses the original dataset that Jeffery shared and trained and tested using a lightGBM model. I have also modified the comparison script that Jeffery made to compare this along with the neural network model and the baseline model.

3. I have added another folder named 'with_lag' which has the script for introducing the lag and also training and testing using the lightGBM model, however, I don't think the lag adds any reasonable value to the result.
