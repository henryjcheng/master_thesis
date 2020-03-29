# script to load sql files that alters the schema for tables in mimic.db

sqlite3 ".database" \
    ".open "../../database/mimic.db"" \
    ".read alter_table.sql" \
    ".table" \
    ".schema d_cpt" \
    ".quit"

