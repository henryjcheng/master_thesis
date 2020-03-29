# script to create mimic database using sqlite
# need to download the csv files first and unzip
# website: https://physionet.org/content/mimiciii/1.4/

mkdir ../../database

sqlite3 ../../database/mimic.db \
    ".mode csv" \
    ".import "../../data/mimic/D_CPT.csv" d_cpt" \
    ".import "../../data/mimic/DIAGNOSES_ICD.csv" diagnoses_icd" \
    ".import "../../data/mimic/D_ICD_DIAGNOSES.csv" d_icd_diagnoses" \
    ".import "../../data/mimic/D_ICD_PROCEDURES.csv" d_icd_procedures" \
    ".import "../../data/mimic/D_ITEMS.csv" d_items" \
    ".import "../../data/mimic/D_LABITEMS.csv" d_labitems" \
    ".import "../../data/mimic/NOTEEVENTS.csv" noteevents" \
    ".import "../../data/mimic/PATIENTS.csv" patients" \
    ".import "../../data/mimic/PRESCRIPTIONS.csv" prescriptions" \
    ".import "../../data/mimic/PROCEDURES_ICD.csv" procedures_icd" \
    ".table" \
    ".quit"
