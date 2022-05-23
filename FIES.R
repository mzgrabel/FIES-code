# change working directory to where the data file is
setwd("C:\\Users\\mzgra\\Documents\\Research2022")
getwd()

#increase memory limit size to handle big data
memory.limit(size = 100000000000)

FIES <- read.csv('analysisData_withFIES.csv')

dim(FIES)
colnames(FIES)
sapply(FIES, class)

FIES$Sex <- ifelse(FIES$Gender == 'M', 0, 1) # 0 = M 1 = F
FIES <- FIES[,-7] # remove gender
FIES$Birth_cohort <- ifelse(FIES$Birth_cohort == "<1981", 1, ifelse(FIES$Birth_cohort == "1981 - 1988", 2, ifelse(FIES$Birth_cohort == "1989 - 1994", 3, ifelse(FIES$Birth_cohort == "1995 - 1998", 4, ifelse(FIES$Birth_cohort == "1999 - 2005", 5, 6)))))
FIES <- FIES[complete.cases(FIES),]
dim(FIES)
head(FIES)


library(writexl)
write_xlsx(FIES,"C:\\Users\\mzgra\\Documents\\Research2022\\FIES.xlsx")
