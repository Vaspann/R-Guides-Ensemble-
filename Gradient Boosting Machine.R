#Gradient boosting machines (GBMs) are an extremely popular machine 
#learning algorithm that have proven successful across many domains 
#GBMs build an ensemble of shallow trees in sequence with each tree 
#learning and improving on the previous one. Although shallow trees
#by themselves are rather weak predictive models, they can be “boosted”
#to produce a powerful “committee” that, when appropriately tuned, is often
#hard to beat with other algorithms.

library(caret)
library(caTools)
library(mlbench)


#Load and inspect
data("PimaIndiansDiabetes2")
df <- PimaIndiansDiabetes2
head(df)
str(df)
summary(df)

#Omit NAs
df <- na.omit(df)


#The anlysis on this data has already been done in previous repos
#We will only implement and tune our gbm model using caret package


set.seed(123)  # for reproducibility

splitcc <- sample.split(df, SplitRatio = 0.7)
train <- subset(df, splitcc == "TRUE")
test <- subset(df, splitcc == "FALSE")

#Minimum Accuracy

table(train$diabetes)
minimum <- 170/260

#trainControl 

trainCtrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

#default gbm model
#use bernoulli for classification

gbm_default <- train(diabetes~.,
                          data = train,
                          method = "gbm",
                          distribution = "bernoulli",
                          trControl = trainCtrl,
                          verbose = FALSE)

gbm_default

#Model Accuracy % 81.54
p <- predict(gbm_default, train)
confusionMatrix(train$diabetes, p)


#parameters of gbm
getModelInfo()$gbm$parameters



#Tuning gbm parameters to optimise our model

hyper_grid <- expand.grid(
  n.trees = c(50,75,100,125,150),
  shrinkage = c(0.075, 0.1,0.125,0.15,0.2),
  interaction.depth = c(1, 3, 5, 7, 9),
  n.minobsinnode = c(7, 10,12, 15,17)
)


gbm_tune <- train(diabetes~.,
                       data = train,
                       method = "gbm",
                       distribution = "bernoulli",
                       trControl = trainCtrl,
                       verbose = FALSE,
                       tuneGrid = hyper_grid )


#best gbm model

gbm_tune$bestTune


#final model

final_grid <- gbm_tune$bestTune


gbm_final <- train(diabetes~.,
                        data = train,
                        method = "gbm",
                        trControl = trainCtrl,
                        verbose = FALSE,
                        tuneGrid = final_grid)

#Accuracy % 83.46
p <- predict(gbm_final, train)
confusionMatrix(train$diabetes, p)



#Implement model on test set

#Minimum accuracy to beat
table(test$diabetes)
test_minimum <- 92/132

# Test Accuracy % 81.82 
predicted <- predict(gbm_final, newdata = test)
confusionMatrix(test$diabetes, predicted)




