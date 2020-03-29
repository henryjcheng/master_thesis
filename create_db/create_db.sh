# script to create mimic database using sqlite
# need to download the csv files first and unzip
# website: https://physionet.org/content/mimiciii/1.4/

# follow this solution for creating tables with correct schema
# https://stackoverflow.com/questions/20240315/how-to-import-csv-file-to-sqlite-with-correct-data-types

rm -rf ../../database
mkdir ../../database

sqlite3 ../../database/mimic.db \
    ".mode csv" \
    ".import "../../data/mimic/D_CPT.csv" d_cpt" \
    ".table" \
    ".quit"
