# Query from Google Cloud Platform
# against MIMIC-III dataset

install.packages("bigrquery")
library(bigrquery)

projectID <- "mimic-iii-260800"

sql <-	"SELECT * FROM `physionet-data.mimiciii_notes.noteevents` limit 5"

notes <- query_exec(sql, project = projectID, use_legacy_sql = F, max_pages = Inf)


# Tokenizer
install.packages("tokenizers")
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

