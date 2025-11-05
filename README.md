# fall-2025-climate-change-and-hydropower
Team project: fall-2025-climate-change-and-hydropower

We do a 80:10:10 split. 90% is int train_val_set.csv and 10% is in final_test_set.csv We do not touch this until the very end.

Features:

year - Year of row

month - Month of row

RectifHyd_MWh - Our Target, the MWh (Megawatt hour) the dam produced. RectifHyd is our source for this data.

mode - Either "Storage" or "RoR" indicating if the plant is primarily operated as a storage or ron-of-river facility

Latitude - Coordinate

Longitude - Coordinate

nerc_region -  Four letter code for the NERC region of the facility

Primary Purpose - What the dam is used for primarily

Dam Height (Ft) - Height of the dam, in feet to the nearest foot, which is defined as the difference between the lowest elevation on the crest of the dam and the lowest elevation in the original streambed; or if not present, the lowest elevation of the downstream toe of the embankment.

NID Height (Ft) - Maximum value of either the dam height, structural height, or hydraulic height. Accepted as the general height of the dam.

Dam Length (Ft) - Length of the dam, in feet, which is defined as the length along the top of the dam. This also includes the spillway, powerplant, navigation lock, fish pass, etc., where these form part of the length of the dam. If detached from the dam, these structures should not be included.

Year Completed - Year (four digits) when the original main dam structure was completed. If unknown, and a reasonable estimate is unavailable, 0000 is used.

NID Storage (Acre-Ft) - Maximum value of normal storage and maximum storage. Accepted as the general storage of the dam.

Max Storage (Acre-Ft) - Maximum storage is defined as the total storage space in a reservoir below the maximum attainable water level, including any surcharge storage

Normal Storage (Acre-Ft) - Normal storage is defined as the total storage space in a reservoir below the normal retention level, including dead and inactive storage and excluding any flood control or surcharge storage. For normally dry flood control dams, the normal storage will be a zero value. If unknown, enter “UNK” and not zero.

Surface Area (Acres) - Surface area, in acres, of the impoundment at its normal water level.

Drainage Area (Sq Miles) - Drainage area of the dam, in square miles, which is defined as the area that drains to the dam reservoir(s).

Division_ID - Division code given by the NCEI that tells us what state and climate region this dam is in

tmin - Minimum temperature this month (deg. F. to 10ths)

tmax - Maximum temperature this month (deg. F. to 10ths)

tavg - Average temperature this month (deg. F. to 10ths)

pcpn - Precipitation this month (inches to 100ths)

The Feature definitions are as per [link](https://floridadep.gov/sites/default/files/Dam%20Parameter%20Definitions.pdf)
The Dams dataset is from the National Inventory of Dams (NID) [link](https://nid.sec.usace.army.mil/nid/#/)
The weather dataset is from the National Centers for Environmental Information (NCEI) https://www.ncei.noaa.gov/pub/data/cirs/climdiv/
 
