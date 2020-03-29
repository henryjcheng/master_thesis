# Create MIMIC Database
## Initialization/Load csv file
First, create a SQLite database and load the csv into the database by running in terminal:
```shell script
./create_db.sh
```
SQLite assumes all fields are TEXT when import csv so need to do some additional work to ensure the correct data type.  
[Referencing this stackoverflow solution](https://stackoverflow.com/questions/20240315/how-to-import-csv-file-to-sqlite-with-correct-data-types)

