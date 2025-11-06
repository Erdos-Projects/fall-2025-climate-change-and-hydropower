### Climate Change and Hydropower 

This is a repository for the Climate Change and Hydropower project as part of The Erdős Institute Fall 2025 Data Science Bootcamp.
Team members: [Daniela Beckelhymer](https://github.com/dbeck7296), [Jeffrey Jackson](https://github.com/jeffreyjacksonwork), [Cory Peters](https://github.com/CPeters31415926535), [Sreelakshmi Sindhu](https://github.com/sreelakshmisindhu).

## Table of contents

1. [Introduction](## Introduction)
2. [Dataset](## Dataset)
3. [Input features](## Input features)
4. [Methods and Model](## Methods and Model)
5. [KPI and Results](## KPI & Results)
6. [Future Directions](## Future Directions)
7. [Sources](## Sources)


## Introduction

Hydropower supplies roughly 30-35% of US renewable electricity and plays a vital role in balancing the power grid, especially as variable solar and wind generation expand. Hydroelectric output is highly sensitive to climatic and hydrologic variability, especially precipitation, temperature, snowpack, and river flow, as well as to the physical design and storage capacity of individual dams. Studies show that climate anomalies such as droughts, heat waves, changing snowmelt timing already affect hydroelectric generation across the US regions. Despite these findings, there is no unified, data driven framework that links nationwide dam characteristics and observed climate data to predict hydropower generation outcomes on a monthly basis.

This project addresses this gap by integrating the important climate record and hydroplant meta-data to build a reproducible model that estimates monthly hydropower generation from 2001-2021 at the dam level based on physical and climate data. By quantifying how physical design and climate variables jointly explain production variability, the model supports federal agencies (DOE, USACE, NOAA) that manage water and energy, regional grid operators (NERC, ISOs) that ensure reliability and utility companies that plan operations and market participation. It also supports researchers and policy makers who assess climate impacts, economic risks and ecosystem tradeoffs tied to hydropower variability.

## Dataset
We collected data for both hydropower output, and climate conditions from 2001-2022.
Climate Data - We used data from the National Centers for Environmental Information (NCEI). The data we collected was monthly precipitation (inches)  and minimum, average, and maximum monthly temperature (fahrenheit). The way the NCEI collects climate information is as follows. First they took the continental US, took each state and for a given state, divided them in up to 10 climate divisions based on geographical and climate factors. They then collect and display climate information they gather from these zones. We use the coordinates from the dams to match which climate division they correspond to. 
Hydropower Data - The power is measured in MWh, and provided by Oak Ridge National Laboratory and Pacific Northwest National Laboratory, under the dataset named RectifHyd. The laboratories took MWh annual measurements and utilized riverflow information to gather the estimated monthly MWh output (hence the name “rectified hydropower”). The dataset included a total of around 1500 dams in the continental US. 
Hydroplant Metadata - We find data about each dam from two sources, the National Inventory of Dams (NID) and from a group named GODEEEP. We include features about the physical aspects of the dam, such as the height and surface area, as well as the primary purpose of the dams, whether it be for hydropower or marine life passage.  

## Input Features

Input features included in the dataset and the sources are listed here.

1. year - Year of row
2. month - Month of row
3. RectifHyd_MWh - Our Target, the MWh (Megawatt hour) the dam produced. RectifHyd is our source for this data.
4. Mode - Either "Storage" or "RoR" indicating if the plant is primarily operated as a storage or ron-of-river facility
5. Latitude - Coordinate
6. Longitude - Coordinate
7. nerc_region -  Four letter code for the NERC region of the facility, entity that governs the maintenance and standardization of the dams.
8. Primary Purpose - What the dam is used for primarily
9. Dam Height (Ft) - Height of the dam, in feet to the nearest foot, which is defined as the difference between the lowest elevation on the crest of the dam and the lowest elevation in the original streambed; or if not present, the lowest elevation of the downstream toe of the embankment.
10. NID Height (Ft) - Maximum value of either the dam height, structural height, or hydraulic height. Accepted as the general height of the dam.
11. Dam Length (Ft) - Length of the dam, in feet, which is defined as the length along the top of the dam. This also includes the spillway, powerplant, navigation lock, fish pass, etc., where these form part of the length of the dam. If detached from the dam, these structures should not be included.
12. Year Completed - Year (four digits) when the original main dam structure was completed. If unknown, and a reasonable estimate is unavailable, 0000 is used.
13. NID Storage (Acre-Ft) - Maximum value of normal storage and maximum storage. Accepted as the general storage of the dam.
14. Max Storage (Acre-Ft) - Maximum storage is defined as the total storage space in a reservoir below the maximum attainable water level, including any surcharge storage
15. Normal Storage (Acre-Ft) - Normal storage is defined as the total storage space in a reservoir below the normal retention level, including dead and inactive storage and excluding any flood control or surcharge storage. For normally dry flood control dams, the normal storage will be a zero value. If unknown, enter ?~@~\UNK?~@~] and not zero.
16. Surface Area (Acres) - Surface area, in acres, of the impoundment at its normal water level.
17. Drainage Area (Sq Miles) - Drainage area of the dam, in square miles, which is defined as the area that drains to the dam reservoir(s).
18. Division_ID - Division code given by the NCEI that tells us what state and climate region this dam is in
19. tmin - Minimum temperature this month (deg. F. to 10ths)
20. tmax - Maximum temperature this month (deg. F. to 10ths)
21. tavg - Average temperature this month (deg. F. to 10ths)
22. pcpn - Precipitation this month (inches to 100ths)

## Methods and Model
The main problem in linking this data to the hydropower data was the fact that it turns out that every dam in the US has 2 different ID numbers, a federal ID and an Energy Information Agency ID (EIA_ID). NID used the federal ID and contained latitude and longitude coordinates, while GODEEEP contained EIA_IDs while also having coordinates. We used nearest neighbors to match each federal ID coordinate to its corresponding EIA_ID. Since the hydropower data was matching dams to their EIA_ID, we now have the metadata corresponding to their federal ID. From here, we need to assign what climate division each dam resides in. Utilizing the coordinates of the dam, we map them onto the climate divisions of the NCEI and attach the value onto the dam. Of course, we also append the corresponding climate information per month and year. The NCEI would have the occasional missing value in its database. We accounted for this by taking the average value of the category (for instance, max temperature) over the given month and climate division.  

We split the data into three parts based on year: 2001-2020 as training data, 2021 as testing data, and 2022 as validation data. From this split, we used two different models.
Neural Network - We employed a fully connected feedforward network with one hidden layer. This model identified climate division, dam area, dam height, and purpose (whether the dam has significant water storage or is a run-of-the-river dam with a relatively small reservoir). Utilized pytorch to train the model.
Decision Tree - We also trained a LightGBM decision tree model on our data. This model identified climate division, geographic location, dam area and dam height as the most important predictors for hydropower output.

On the withheld data, our two models performed as follows:
![alt text](https://github.com/Erdos-Projects/fall-2025-climate-change-and-hydropower/blob/main/results/final/test_predictions_comparison.png)

## KPI & Results
We evaluate model performance and practical usefulness through
R^2, measuring the percentage of variation in monthly hydropower generation explained by the model
RMSE and MAE, quantifying the average prediction error in megawatt hours(MWh) between the predicted and observed output.

These metrics assess how well the model captures real-world hydropower behavior across diverse regions and dam types, given a changing climate.

Our neural network achieved an R^2 value of 0.773, whereas the LightGBM model has an R^2 of 0.930. Thus, our decision tree-based approach was the better of the two at predicting novel hydropower. The MAE and RMSE of our LightGBM model were 5,069 and 19,214 MWh respectively.

The most important features to LightGBM are all geographically and physically based, rather than climate related.

![alt text](https://github.com/Erdos-Projects/fall-2025-climate-change-and-hydropower/blob/main/results/final/lightgbm_feature_importance.png)

## Future Directions
To make our results more usable for stakeholders, it would be convenient to make an applet where a user selects a dam and inputs a monthly weather forecast and gets back a hydropower prediction.
One limitation of our model is that it did not take into account the characteristics of the stream itself such as water height, sediment content or river flow speed. The United States Geological Survey (USGS) provides daily data for monitoring location for these parameters. Our model could be made more accurate after incorporating this data. 
Our model currently considers a limited amount of climate data - only temperature and precipitation. Other factors such as snowfall or wind speed could potentially impact the power output of a dam.


## Sources

(1) The Feature definitions are as per [link](https://floridadep.gov/sites/default/files/Dam%20Parameter%20Definitions.pdf)

(2) The Dams dataset is from the National Inventory of Dams (NID) [link](https://nid.sec.usace.army.mil/nid/#/)

(3) The weather/climate dataset is from the National Centers for Environmental Information (NCEI) https://www.ncei.noaa.gov/pub/data/cirs/climdiv/

(4) Metadata for EIA_ID not found in NID (Latitude, Longitude, nerc_region, mode)  dams/hydroplants https://zenodo.org/records/13776945 (godeeep_hydro_plants.csv)

(5) MWh feature target from Oak Ridge National Laboratory and Pacific Northwest National Laboratory https://zenodo.org/records/11584567
