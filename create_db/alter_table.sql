-- script to alter the schema for tables in mimic.db

PRAGMA foreign_keys=off;
BEGIN TRANSACTION;

-- ========== d_cpt ==========
ALTER TABLE d_cpt RENAME TO _d_cpt_original;
CREATE TABLE d_cpt(
  "ROW_ID" INTEGER NOT NULL PRIMARY KEY,
  "CATEGORY" INTEGER,
  "SECTIONRANGE" TEXT,
  "SECTIONHEADER" TEXT,
  "SUBSECTIONRANGE" TEXT,
  "SUBSECTIONHEADER" TEXT,
  "CODESUFFIX" TEXT,
  "MINCODEINSUBSECTION" INTEGER,
  "MAXCODEINSUBSECTION" INTEGER
);
INSERT INTO d_cpt ("ROW_ID", "CATEGORY", "SECTIONRANGE", "SECTIONHEADER",
                   "SUBSECTIONRANGE", "SUBSECTIONHEADER", "CODESUFFIX",
                   "MINCODEINSUBSECTION", "MAXCODEINSUBSECTION")
  SELECT "ROW_ID", "CATEGORY", "SECTIONRANGE", "SECTIONHEADER",
         "SUBSECTIONRANGE", "SUBSECTIONHEADER", "CODESUFFIX",
         "MINCODEINSUBSECTION", "MAXCODEINSUBSECTION"
  FROM _d_cpt_original;
DROP TABLE _d_cpt_original;

-- ========== d_icd_diagnoses ==========
ALTER TABLE d_icd_diagnoses RENAME TO _d_icd_diagnoses_original;
CREATE TABLE d_icd_diagnoses(
  "ROW_ID" INTEGER NOT NULL PRIMARY KEY,
  "ICD9_CODE" TEXT,
  "SHORT_TITLE" TEXT,
  "LONG_TITLE" TEXT,
  FOREIGN KEY ("ICD9_CODE")
    REFERENCES DIAGNOSES_ICD ("ICD9_CODE")
);
INSERT INTO d_icd_diagnoses ("ROW_ID", "ICD9_CODE", "SHORT_TITLE", "LONG_TITLE")
  SELECT "ROW_ID", "ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"
  FROM _d_icd_diagnoses_original;
DROP TABLE _d_icd_diagnoses_original;

-- ========== d_icd_procedures ==========
ALTER TABLE d_icd_procedures RENAME TO _d_icd_procedures_original;
CREATE TABLE d_icd_procedures(
  "ROW_ID" INTEGER NOT NULL PRIMARY KEY,
  "ICD9_CODE" TEXT,
  "SHORT_TITLE" TEXT,
  "LONG_TITLE" TEXT,
  FOREIGN KEY ("ICD9_CODE")
    REFERENCES PROCEDURES_ICD ("ICD9_CODE")
);
INSERT INTO d_icd_procedures ("ROW_ID", "ICD9_CODE", "SHORT_TITLE", "LONG_TITLE")
  SELECT "ROW_ID", "ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"
  FROM _d_icd_procedures_original;
DROP TABLE _d_icd_procedures_original;

-- ========== d_itmes ==========
ALTER TABLE d_items RENAME TO _d_items_original;
CREATE TABLE d_items(
  "ROW_ID" INTEGER NOT NULL PRIMARY KEY,
  "ITEMID" INTEGER,
  "LABEL" TEXT,
  "ABBREVIATION" TEXT,
  "DBSOURCE" TEXT,
  "LINKSTO" TEXT,
  "CATEGORY" TEXT,
  "UNITNAME" TEXT,
  "PARAM_TYPE" TEXT,
  "CONCEPTID" INTEGER
);
INSERT INTO d_items ("ROW_ID", "ITEMID", "LABEL", "ABBREVIATION",
                     "DBSOURCE", "LINKSTO", "CATEGORY", "UNITNAME",
                     "PARAM_TYPE", "CONCEPTID")
  SELECT "ROW_ID", "ITEMID", "LABEL", "ABBREVIATION",
         "DBSOURCE", "LINKSTO", "CATEGORY", "UNITNAME",
         "PARAM_TYPE", "CONCEPTID"
  FROM _d_items_original;
DROP TABLE _d_items_original;

-- ========== diagnoses_icd ==========
ALTER TABLE diagnoses_icd RENAME TO _diagnoses_icd_original;
CREATE TABLE diagnoses_icd(
  "ROW_ID" INTEGER NOT NULL PRIMARY KEY,
  "SUBJECT_ID" INTEGER,
  "HADM_ID" INTEGER,
  "SEQ_NUM" INTEGER,
  "ICD9_CODE" TEXT,
  FOREIGN KEY ("SUBJECT_ID")
    REFERENCES DIAGNOSES_ICD ("SUBJECT_ID")
  FOREIGN KEY ("ICD9_CODE")
    REFERENCES D_ICD_DIAGNOSES ("ICD9_CODE")
);
INSERT INTO diagnoses_icd ("ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE")
  SELECT "ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"
  FROM _diagnoses_icd_original;
DROP TABLE _diagnoses_icd_original;

COMMIT;
PRAGMA foreign_keys=on;