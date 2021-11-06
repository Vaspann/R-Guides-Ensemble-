#xgboost is short for eXtreme Gradient Boosting. It is an efficient and scalable
#implementation of gradient boosting framework by (Friedman, 2001) (Friedman et al., 2000).


library(caret)
library(mlbench)
library(xgboost)


#Load and inspect
data("PimaIndiansDiabetes2")
df <- PimaIndiansDiabetes2
glimpse(df)
summary(df)

#use 0s & 1s for classsification problem
df$diabetes <- as.numeric(df$diabetes) - 1


#Omit NAs
df <- na.omit(df)
table(df$diabetes)


set.seed(123)  # for reproducibility
#split into training (80%) and testing set (20%)
splittt = createDataPartition(df$diabetes, p = .8, list = F)
train = df[splittt, ]
test = df[-splittt, ]


#create matrix
trainlen <- as.matrix(train[,-9])
trainlabel <- train[,"diabetes"]

testlen <- as.matrix(test[,-9])
testlabel <- test[,"diabetes"]

#setting parameters
parameters <- list(set.seed = 123,
                   eval_metric = "error",
                   eval_metric = "logloss",
                   objective = "binary:logistic")

#xgboost model (default eta = 0.3 & max_depth = 6)
model <- xgboost(data = trainlen, label = trainlabel,
                 nrounds = 100,
                 nthread = 3,
                 params = parameters)


#predict on test set
pred <- predict(model, testlen)
pred <- ifelse(pred > 0.5, 1, 0) 

#minimum accuracy % 70.51
table(test$diabetes)
minimum <- 55/78

#confusionmatrix, accuracy % 79.49
confusionMatrix(as.factor(pred), as.factor(testlabel))


#model evaluation
attributes(model)
plot(model$evaluation_log)
xgb.plot.importance(xgb.importance(model = model))
xgb.plot.shap(data = trainlen, 
              model = model,
              top_n = 5)