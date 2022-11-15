# Massive-Data-CS543
Analyzing flight and weather data to build clusters and predictive models


## Setup
Please place the raw csv files for each airplane year under airplane_raw directory
For the metadata, you can place it under airplane_raw/metadata (stuff like airports.csv etc.)

For the weather data, please extract ghcnd_all.tar.gz into a directory under the root called ghcnd_all
Place the weather station metadata file, ghcnd-stations.txt, under weather_raw/metadata

## Processing
In order to join weather data to airplane data, please first run the weather_airline_processing.ipynb file.
This will generate intermediate maps needed to efficiently join the weather daily data to the airplane flight data.

Then, please run the data_processing.ipynb notebook. This will actually join the weather data, add a unique record ID
and then save the data in parquet format in directory ./FINAL_processed_data

We provide our processed data, clusters, errors, and models at the following drive link: 
[here](https://drive.google.com/drive/folders/1gDONjpM9gBYyLcgn1S3Sw5Chu5NICH5E?usp=sharing)


## Batched Execution
Please refer to bfr_clusters.ipynb for an example of batched execution/clustering.
The first few cells are preparing and normalizing the data, while the last cell is the clustering loop 
for each partition.