# Create MIMIC Database using SQLite

follow this solution for creating tables with correct schema
https://stackoverflow.com/questions/20240315/how-to-import-csv-file-to-sqlite-with-correct-data-types
sqlite assumes all fields are TEXT when import csv so need to do some additional work to ensure the correct datatype