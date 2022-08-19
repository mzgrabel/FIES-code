# change working directory to where the data file is
setwd("C:\\Users\\mzgra\\Documents\\Research2022")
getwd()

#increase memory limit size to handle big data
memory.limit(size = 1000000000)


FIES <- read.csv('analysisData_withFIES.csv')
length(unique(FIES$eDWID)) # 27,021 unique patients before
# Remove "CareEpi_EndDt", "CareEpi_StartDt", "encounterdate", "year", "zcta", "FEV1_pct_predicted", "alive", "LT", "tSince_BL", "SESlow_ever", "PA_ever", "isOnEnzymes_ever", "Vx770_ever", "Vx809_ever", "smoking_ever", "smoking_household_ever", "second_smoke_ever", "mrsa_ever" 
FIES <- FIES[,-c(48, 49, 8, 3, 4, 9, 14, 47, 52, 22, 23, 26, 27, 28, 29, 30, 31, 24)]
colnames(FIES)

FIES <- FIES[complete.cases(FIES), ]
length(unique(FIES$eDWID)) # 26,375 unique patients after complete cases


FIES$Sex <- ifelse(FIES$Gender == 'M', 0, 1) # 0 = M 1 = F
FIES <- FIES[,-5]

#unique(FIES$Birth_cohort) # change to 1,2,3,4,5,6 for each cohort starting from earliest year
FIES$Birth_cohort <- ifelse(FIES$Birth_cohort == "<1981", 1, ifelse(FIES$Birth_cohort == "1981 - 1988", 2, ifelse(FIES$Birth_cohort == "1989 - 1994", 3, ifelse(FIES$Birth_cohort == "1995- 1998", 4, ifelse(FIES$Birth_cohort == "1999 - 2005", 5, 6)))))
# head(FIES)
# sum(FIES$eDWID == 900000742)
# 
# m <- c()
# for (i in 1:20153){
#   s <- sum(FIES$eDWID == unique(FIES$eDWID[i]))
#   append(s, m)
# }

library(writexl)
write_xlsx(FIES,"C:\\Users\\mzgra\\Documents\\Research2022\\FIES.xlsx")
