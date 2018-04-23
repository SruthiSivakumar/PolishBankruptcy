# set the workking directory in order to access the required files 
setwd("G:\\R_Studio\\R_Workspace\\PolishBankruptcy_Ex-4")
getwd()

#Install and load all the libraries needed to build the model 
library(rJava) #To Work with arff files
library(RWeka) #To work with arff files
library(ggplot2) # Visualization purposes
library(caret)
library(e1071)
library(corrplot) #To generate a correlation Plot
library(missForest) # To use random forest Imputaion
library(FNN) # to use KNN Imputation
library(reshape)
library(reshape2)
library("C50") # TO USE Decision Tree
library(gmodels) #Required to create a confution matrix

# Reading the first file, take a peek into the file and 
# insert the years column before the class variable
# Optionally you can save it as a CSV file also
# Then repeat the same for all the 5 files

#First Year 
pol_bank_data_1 <- read.arff("1year.arff")
#names(pol_bank_data_1)
#summary(pol_bank_data_1)
years <- unlist(rep(5, NROW(pol_bank_data_1)))
pol_bank_data_1 <- cbind(pol_bank_data_1, years)
pol_bank_data_1 <-pol_bank_data_1[, c(1:64,66,65)]
head(pol_bank_data)

# Second Year
pol_bank_data_2 <- read.arff("2year.arff")
years <- unlist(rep(4, NROW(pol_bank_data_2)))
pol_bank_data_2 <- cbind(pol_bank_data_2, years)
pol_bank_data_2 <-pol_bank_data_2[, c(1:64,66,65)]
head(pol_bank_data)

#Third Year
pol_bank_data_3 <- read.arff("3year.arff")
years <- unlist(rep(3, NROW(pol_bank_data_3)))
pol_bank_data_3 <- cbind(pol_bank_data_3, years)
pol_bank_data_3 <-pol_bank_data_3[, c(1:64,66,65)]
head(pol_bank_data)

#Fourth Year
pol_bank_data_4 <- read.arff("4year.arff")
years <- unlist(rep(2, NROW(pol_bank_data_4)))
pol_bank_data_4 <- cbind(pol_bank_data_4, years)
pol_bank_data_4 <-pol_bank_data_4[, c(1:64,66,65)]
head(pol_bank_data)

#Fifth Year
pol_bank_data_5 <- read.arff("5year.arff")
years <- unlist(rep(1, NROW(pol_bank_data_5)))
pol_bank_data_5 <- cbind(pol_bank_data_5, years)
pol_bank_data_5 <-pol_bank_data_5[, c(1:64,66,65)]
head(pol_bank_data)

# Now combine all the 5 years data into one single dataset that we can work on
pol_bank_data <- rbind(pol_bank_data_1,pol_bank_data_2,pol_bank_data_3,
                       pol_bank_data_4,pol_bank_data_5)


# Taking a look at how the data is
summary(pol_bank_data)
write.csv(pol_bank_data,"Combined_Data.csv")

# It seems that there are NA's in all the columns and all the variables are continous
# Taking a look at how the class varialbe is distributed.
table(pol_bank_data$class)
# we find that the values are highly imbalanced
# We might later use the different Sampling techniques to avoid such imbalance
# On analysing the data we can infer few things about the data
# The below column has the least no of NA.
summary(pol_bank_data$Attr55)

# The below column has the most no of NA. 
# Imputing more than 43% of the whole data will not make sense, hence we remove the column
summary(pol_bank_data$Attr37)

# As a first step we remove all the rows that contains NA and 
# take a look at the correlation between the independent variable to help with feature selection
pol_new <- pol_bank_data[complete.cases(pol_bank_data[,-37]), ]

summary(pol_new)

table(pol_new$class)
prop.table(pol_bank_data$class)

cormat_new <- cor(pol_new[,-65])
write.csv(cormat, "correlation_Matrix.csv")

# Let us try to impute the data in other ways so that we can compare with the above dataset

summary(pol_bank_data)


melt_data <- melt(pol_bank_data[,], id.var = "class")

p<- ggplot(data = melt_data, aes(x= variable, y=value)) +
  geom_boxplot(aes(fill = class))
p <- p +facet_wrap( ~ variable, scales = 'free')
p

# Making correlation matrix for with all the cases
# cormat_all <- cor(pol_bank_data, complete.cases())
Pol_all <- pol_bank_data[complete.cases(pol_bank_data), ]

cormat_all <- cor(Pol_all[,-66])
# plottig the correlation plot
par(mfrow=c(1,2))
plot.new()
corrplot(cormat_all)
corrplot(cormat_new)
dev.off()

#I hope removing nas as such does not help much.
# Lets us try to impute in different methods after studying the data furtehr

# Removing the Column that had most number of NA
pol_bank_data <- pol_bank_data[, -c(37)]
# Removing NA's wrt specific columns alone.
summary(pol_bank_data$Attr29)
naomit <- complete.cases(pol_bank_data[, "Attr29"])
pol_bank_data<- pol_bank_data[naomit, ]
summary(pol_bank_data)

naomit <- complete.cases(pol_bank_data[, "Attr13"])
pol_bank_data<- pol_bank_data[naomit, ]
summary(pol_bank_data)

# As Attr29 is the log of the actual value, let us convert it to the actual value, which may help us.
pol_bank_data$Attr29 <- exp(pol_bank_data$Attr29)
summary(pol_bank_data$Attr29)

# Removing repeated ratios as they tend to create multicollinearity
pol_bank_data <- pol_bank_data[, -c(15,17,20,62)]

# as the usual, mean/ mode imputation might change the actual values of these ratios,
# We directly apply the decision tree on this dataset
set.seed(60)
split = createDataPartition(pol_bank_data$class, p = 0.70, list = FALSE, times = 1)
pol_bank_train <- pol_bank_data[split,]
pol_bank_test <- pol_bank_data[-split,]

# the 61st column in the training dataset is default class variable
# which we will be supply as the target factor vectr for classification

pol_bank_model <- C50::C5.0(pol_bank_train[,-61], pol_bank_train$class)
summary(pol_bank_model)   

"Attribute usage:
100.00%	Attr35  87.56%	Attr46 65.55%	Attr26 56.13%	Attr27  
37.35%	Attr34  29.94%	Attr24  26.53%	Attr5 14.04%	Attr56
13.64%	years 13.43%	Attr61  7.70%	Attr6 4.29%	Attr44
1.95%	Attr13  1.92%	Attr62  1.81%	Attr60  1.29%	Attr11
0.94%	Attr58  0.86%	Attr39  0.83%	Attr43  0.76%	Attr23
0.71%	Attr21  0.52%	Attr51  0.41%	Attr55  0.36%	Attr40
0.35%	Attr28  0.26%	Attr7 0.25%	Attr33  0.22%	Attr64
0.14%	Attr32  0.13%	Attr3 0.07%	Attr2 "

# prediction
pol_pred <- predict(pol_bank_model, pol_bank_test)

# Confusion Matrix
CrossTable(pol_bank_test$class, pol_pred, prop.chisq = FALSE, prop.r=FALSE,
           prop.c=FALSE, dnn=c("Actual Class", "Predicted Class"))

# Boosting the model

pol_bank_model_boost <- C50::C5.0(pol_bank_train[,-61], pol_bank_train$class, 
                                  trials = 10)
"
Attribute usage:

100.00%	Attr3 100.00%	Attr6 100.00%	Attr25  100.00%	Attr29
100.00%	Attr31  100.00%	Attr35  99.81%	Attr26  99.74%	Attr33
99.69%	Attr46  99.60%	Attr5 99.46%	Attr39  99.43%	Attr49
99.21%	Attr58  98.26%	Attr36  97.50%	Attr34  97.49%	Attr48
97.16%	Attr38  96.51%	Attr61  96.28%	Attr24  96.15%	Attr19
94.24%	Attr62  92.63%	Attr27  91.11%	Attr56  89.07%	Attr50
88.82%	Attr1   88.52%	Attr43  87.81%	Attr44  86.34%	Attr42
86.15%	Attr22  86.06%	Attr4   85.28%	Attr40  84.80%	Attr16
84.30%	Attr23  83.14%	Attr13  82.59%	Attr9   81.72%	Attr55
81.47%	Attr12  81.10%	Attr59  80.63%	Attr7   76.34%	Attr41
73.11%	Attr30  70.92%	Attr8   66.38%	Attr47  65.61%	Attr52
63.45%	Attr51  63.36%	Attr54  57.86%	Attr10  57.20%	Attr11
55.11%	Attr32  42.39%	Attr2   39.23%	Attr57 34.15%	Attr45
27.86%	Attr64  26.91%	years   26.13%	Attr53  9.35%	Attr28
9.24%	Attr60    4.50%	Attr21 "
summary(pol_bank_model_boost)

# prediction
pol_pred_boost <- predict(pol_bank_model_boost, pol_bank_test)

# Confusion Matrix

CrossTable(pol_bank_test$class, pol_pred_boost, prop.chisq = FALSE, prop.r=FALSE,
           prop.c=FALSE, dnn=c("Actual Class", "Predicted Class"))


# accuracy is 96%