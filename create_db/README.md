# Create MIMIC Database
Run following script to create mimic database using SQLite: 
```shell script
./create_db.sh
```
### Data Type Issue
SQLite assumes all fields are TEXT when import from csv.  
[Following this Stackoverflow solution](https://stackoverflow.com/questions/20240315/how-to-import-csv-file-to-sqlite-with-correct-data-types), we handle this issue by:  
1. import all the csv table
2. create the corresponding tables with correct schema
3. copy the records into tables with correct schema
4. remove original tables

