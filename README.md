# USAT
Assisting USAT with ETL and Analysis of Youth and Junior Elite Race Data.

General Flow of analysis (Python+Tableau):
1) Selenium webscraper gathers publicly available data from USAT site
2) Data structure is transformed into long format, with one row represents one Date->Race->Athlete->Leg of Race (eg Swim, Bike, or Run)
3) Data is cleaned, wildly out of pattern results are flagged as invalid
4) Many statistics are computed, the most important being a comparison of each racer's race leg performance to the average of the fastest Top3 racers within that race leg.  The reason this is important is because absolute times are meaningless to compare across races due to wide variation in race course length.
5) A Decision Tree Regressor Model is trained on inputs and the model expectations are added to the data
6) Tableau viz illustrates trends by athlete, team, age, leg of race, and event.

