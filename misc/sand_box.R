# Query from Google Cloud Platform
# against MIMIC-III dataset

# ---- link to GCP ----
install.packages("bigrquery")
library(bigrquery)
library(dplyr)

projectID <- "mimic-iii-260800"

sql <-	"SELECT * FROM `physionet-data.mimiciii_notes.noteevents` limit 5"
notes <- query_exec(sql, project = projectID, use_legacy_sql = F, max_pages = Inf)


sql <- "SELECT * FROM `physionet-data.mimiciii_clinical.chartevents` limit 5"
chart_event <- query_exec(sql, project = projectID, use_legacy_sql = F, max_pages = Inf)

sql <- "SELECT * FROM `physionet-data.mimiciii_clinical.diagnoses_icd` limit 5"
diag <- query_exec(sql, project = projectID, use_legacy_sql = F, max_pages = Inf)

sql <- "SELECT icd9_code, count(*) as cnt FROM `physionet-data.mimiciii_clinical.diagnoses_icd` GROUP BY icd9_code"
diag_count <- query_exec(sql, project = projectID, use_legacy_sql = F, max_pages = Inf)
diag_count <- diag_count %>% arrange(desc(cnt))
summary(diag_count$cnt)

sql <- "SELECT  a.hadm_id, 
                a.text, 
                b.icd9_code 
        FROM `physionet-data.mimiciii_notes.noteevents` AS a
        INNER JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` AS b USING(hadm_id) 
        WHERE b.icd9_code = '4019' LIMIT 100"
notes_icd <- query_exec(sql, project = projectID, use_legacy_sql = F, max_pages = Inf)

sql <- "SELECT  a.hadm_id, 
                a.text, 
                b.icd9_code 
        FROM `physionet-data.mimiciii_notes.noteevents` AS a
        INNER JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` AS b USING(hadm_id) 
        WHERE b.icd9_code = '4019'"
#pos_case <- query_exec(sql, project = projectID, use_legacy_sql = F, max_pages = Inf)
pos_case <- bq_project_query(x=projectID, query=sql)
bq_table_fields(pos_case) 
pos_case <- bq_table_download(pos_case)

# get all the 4019 - hypertension text
# create binary BERT model

# ---- get word freq ----
# Tokenizer
#install.packages("tokenizers")
library(tokenizers)

note_word <- tokenize_words(notes$TEXT[4])

text <- notes$TEXT[5]
nchar(text)
text_sub <- substr(text, 0, 50000)

test <- data.frame(tokenize_words(text))
names(test) <- "test"
test

library(dplyr)
word_freq <- data.frame(test %>% group_by(test) %>% summarise(count=n()))
table(word_freq$count)

# ---- Link text with dx ----

# peek admission table
sql <- "SELECT * FROM `physionet-data.mimiciii_clinical.admissions` limit 5"

admission <- bq_project_query(x=projectID, query=sql)
#bq_table_fields(admission) 
admission <- bq_table_download(admission)

sql <- "SELECT * FROM `physionet-data.mimiciii_clinical.diagnoses_icd` limit 5"

diagnoses_icd <- bq_project_query(x=projectID, query=sql)
diagnoses_icd <- bq_table_download(diagnoses_icd)

# get all notes
sql <- "SELECT  HADM_ID, 
                TEXT
        FROM    `physionet-data.mimiciii_notes.noteevents` 
        WHERE   HADM_ID is not null"

notes_all <- bq_project_query(x=projectID, query=sql)
notes_all <- bq_table_download(notes_all)

write.csv(notes_all, file="C:/Users/Henry/Desktop/Main/School/Master Thesis/Dataset/notes_all.csv",
          row.names=FALSE)

# get all diagnoses
sql <- "SELECT  HADM_ID, 
                ICD9_CODE
        FROM    `physionet-data.mimiciii_clinical.diagnoses_icd` 
        WHERE   HADM_ID is not null"

diagnoes_icd_all <- bq_project_query(x=projectID, query=sql)
diagnoes_icd_all <- bq_table_download(diagnoes_icd_all)

#write.csv(diagnoes_icd_all, file="C:/Users/Henry/Desktop/Main/School/Master Thesis/Dataset/diagnoses_icd_all.csv",
#          row.names=FALSE)

# join notes with diagnoses
notes_diag <- notes_all %>% inner_join(diagnoes_icd_all, by = "HADM_ID")
write.csv(notes_diag, file="C:/Users/Henry/Desktop/Main/School/Master Thesis/Dataset/text_dx.csv",
          row.names=FALSE)

# alternatively, using one sql query, but this cause error because return size too large
sql <- "SELECT  a.TEXT, 
                b.ICD9_CODE 
        FROM    `physionet-data.mimiciii_notes.noteevents` as a
        LEFT JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` as b
                ON a.HADM_ID = b.HADM_ID
        WHERE   a.HADM_ID is not null"

text_dx <- bq_project_query(x=projectID, query=sql)
text_dx <- bq_table_download(text_dx)

write.csv(notes_all, file="C:/Users/Henry/Desktop/Main/School/Class/Stats 404/github_homework/HW2/text_dx.csv",
          row.names=FALSE)

#write.csv(notes_diag, file="C:/Users/Henry/Desktop/Main/School/Master Thesis/Dataset/notes_diag.csv",
#          row.names=FALSE)

# get all diagnosis description
sql <- "SELECT  ICD9_CODE, 
                SHORT_TITLE, 
                LONG_TITLE
        FROM    `physionet-data.mimiciii_clinical.d_icd_diagnoses` 
        WHERE   ICD9_CODE is not null"

diag_desc <- bq_project_query(x=projectID, query=sql)
diag_desc <- bq_table_download(diag_desc)

write.csv(diag_desc[, c(1, 3)], file="C:/Users/Henry/Desktop/Main/School/Master Thesis/Dataset/diag_desc.csv",
          row.names=FALSE)

notes_diag <- notes_diag %>% left_join(diag_desc, by = "ICD9_CODE") %>% 
                             select(HADM_ID, TEXT, ICD9_CODE, LONG_TITLE)

