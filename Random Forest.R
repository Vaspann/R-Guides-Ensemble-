#This method of machine learning combines various trees to produce one optimal predictive model.
#They are usually better than decision tree models as they are combining different models 
#to achieve higher accuracy and to avoid overfitting.

#Load the packages
library(randomForest)
library(mlbench)
library(tidyverse)
library(caTools)
library(caret)

#Load and inspect
data("PimaIndiansDiabetes2")
df <- PimaIndiansDiabetes2
head(df)
str(df)
summary(df)

#Omit NAs
df_2 <- na.omit(df)

# Transforming Features  ----------------------------------------------------
#age, pregnant and inuslin have unequal distributions
#we already analysed the features of this dataset in regression repository

summary(df_2$age)
df_2$age_cap <- as.factor(ifelse(df_2$age<=30, "20-30",
                                  ifelse(df_2$age<=40,"31-40",
                                         ifelse(df_2$age<=50, "41-50", "50+"))))


summary(df_2$pregnant)
df_2$pregnant_cap <- as.factor(ifelse(df_2$pregnant==0, "0",
                                       ifelse(df_2$pregnant<=2, "1-2",
                                              ifelse(df_2$pregnant<=5,"3-5",
                                                     ifelse(df_2$pregnant<=10,"6-10", "10+")))))

summary(df_2$insulin)
df_2$insulin_cap <- as.factor(ifelse(df_2$insulin<=75, "0-75",
                                      ifelse(df_2$insulin<=150, "76-150",
                                             ifelse(df_2$insulin<=180,"151-180",
                                                    ifelse(df_2$insulin<=400,"181-400",
                                                           ifelse(df_2$insulin<=600, "401-600", "600+"))))))



df_2[,c("age","insulin","pregnant")] <- NULL


#Check for equal distibutions
table(df_2$diabetes,df_2$insulin_cap)
table(df_2$diabetes,df_2$pregnant_cap)
table(df_2$diabetes, df_2$age_cap)



# Train and test set ------------------------------------------------------

splitcc <- sample.split(df_2, SplitRatio = 0.7)
train <- subset(df_2, splitcc == "TRUE")
test <- subset(df_2, splitcc == "FALSE")


# Testing models Random Forest --------------------------------------------

set.seed(123)
model <- randomForest(diabetes~., data = train, proximity = TRUE)
model


#Plot the error rates to see if default 500 trees is enough for optimal classification

oob_error_rate <- data.frame(
  Trees = rep(1:nrow(model$err.rate),times = 3),
  Type = rep(c("OOB","neg", "pos"),each=nrow(model$err.rate)),
  Error = c(model$err.rate[,"OOB"],
            model$err.rate[,"neg"],
            model$err.rate[,"pos"]))

ggplot(oob_error_rate, aes(x = Trees, y = Error)) +
  geom_line(aes(color = Type))


#add more trees to see if error rate decreases
#in this exmaple it does

model_2 <- randomForest(diabetes~., data = train, ntree = 1000, proximity = TRUE)
model_2

oob_error_rate_2 <- data.frame(
  Trees = rep(1:nrow(model_2$err.rate),times = 3),
  Type = rep(c("OOB","neg", "pos"),each=nrow(model_2$err.rate)),
  Error = c(model_2$err.rate[,"OOB"],
            model_2$err.rate[,"neg"],
            model_2$err.rate[,"pos"]))

#The error rate stabilises after roughly 900 trees
ggplot(oob_error_rate_2, aes(x = Trees, y = Error)) +
  geom_line(aes(color = Type))


#Find the optimal number of variables tried at each split by creating a vector
#that stores the error rate for different models wit different mtry values 
#up to 8 since we have 8 features 

oob_values<- vector(length = 8)

for (i in 1:8) {
  A <- randomForest(diabetes ~., data = train, mtry = i, ntree = 1000)
  oob_values[i] <- A$err.rate[nrow(A$err.rate), 1]
}

#The optimal value for mtry was the default value  mtry = 2
#which had the lowest OOB rate

oob_values


# Test set ----------------------------------------------------------------

#minimum accuracy to beat % 66.8

table(df_2$diabetes)
minimum <- 262/392

#Implement the optimal model using mtry = 2 and ntree = 1000 on the test set

set.seed(123)
model_3 <- randomForest(diabetes~., data = test, mtry = 2, ntree = 1000)
model_3

#Our model achieved a % 75.57 accuracy
