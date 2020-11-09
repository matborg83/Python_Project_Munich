# Rent Prices & Nearby Venues Data Analysis of Munich

## Project in progress

## Introduction

Munich is the third-largest city in Germany with about 1.56 million inhabitant. It is divided into 25 districts  and it has the most populated density per km2 in Germany (4,500 people per km2 ). Together with London, Paris and Berlin, the Munich property market ranks as one of the most important in Europe. As in 2020 demand exceeded available property, Munich remain Europe's favourite locations among investors [1].  

In this report, it is expected that investors, consider, among others, the density of social places  and the average price for monthly rents for the district. Based on these two factors, Munich’s district will be segmented and clustered using Python and k-means machine learning. The report will present summary tables and the clustered districts plotted on a map so investors can make better informed decisions. 


## Data section 

Webscrapping 
The report contains the following data:

•	 “Bayern Postal Codes” from geonames.org[2] using Pandas library webscrapping function panda.read_html.

•	 “Average rent of apartments in Munich, Germany in the first half of 2019, by district”[3] Excel CSV document downloaded from Statista and imported into Github project’s repository.

•	 “Postal Codes in Munich” [4] from official Munich website imported into Github project’s repository.

•	Foursquare locations API to collect all venues in Munich

Dataset 2 in Notebook contains combined dataset temp_1 and temp_2 (“Average rent of apartments in Munich…”  and “Postal Codes in Munich”). Ten districts were missing in temp_1 and will be dropped in temp_2 (example: Moosach, Obergiesing). As a result 78 Postal code out of 128 contain rent average prices.
We could have merge the monthly rent price from "Munich" to the 10 missing districts but it would be rather inaccurate to do so. Instead beautiful soup could be used to scrap apartment websites in Munich and find the missing average rents for the missing districts.


References:

[1] https://www.muenchen.de/rathaus/wirtschaft_en/munich-business-location/economic-data.html

[2] https://www.geonames.org/postal-codes/DE/BY/bayern.html

[3] https://www.statista.com/statistics/800552/rental-costs-in-munich-germany-by-district/

[4] https://www.muenchen.de/int/en/living/postal-codes.html

