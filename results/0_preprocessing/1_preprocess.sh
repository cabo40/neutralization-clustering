#!/bin/sh

#Removes unused columns
cut --complement -d "," -f 3,6-9 crsp_raw.csv > crsp_cut.csv
#Removes empty data
awk -F',' -f crsp_preproc.awk crsp_cut.csv > crsp_preproc.csv
rm crsp_cut.csv
#Removes trailing comma
awk -F',' -f crsp.awk crsp_preproc.csv | sed 's/,$//' > crsp_big.csv
rm crsp_preproc.csv
#Removes repeated data
awk -F',' -f crsp_trim.awk crsp_big.csv > crsp_small.csv
rm crsp_big.csv
