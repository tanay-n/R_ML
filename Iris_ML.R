
##### IRIS DATASET - IMPLEMENTING MACHINE LEARNING ALGORITHMs #####

##################################################################################################################
## check for installed packages
if (("caret" %in% rownames(installed.packages())) == FALSE) {
        install.packages("caret", dependencies = c("Depends", "Suggests")) #install required package
}

if (("ellipse" %in% rownames(installed.packages())) == FALSE) {
        install.packages("ellipse", dependencies = c("Depends", "Suggests")) #install required package
}


## additional checks due to errors for package 'caret'
if (("ModelMetrics" %in% rownames(installed.packages())) == FALSE) {
        install.packages("ModelMetrics", dependencies = c("Depends", "Suggests")) #install required package
}

if (("gower" %in% rownames(installed.packages())) == FALSE) {
        install.packages("gower", dependencies = c("Depends", "Suggests")) #install required package
}

## importing required packages
library("caret")
library("ellipse")


##################################################################################################################
## loading dataset
data(iris)
dataset <- iris


## exploring dataset
str(dataset)
summary(dataset)


## creating validation datasets 
validation_index <- createDataPartition(dataset$Species, p = 0.8, list = FALSE)
train_set <- dataset[validation_index, ]
test_set <- dataset[-validation_index, ]


## dimensions of the dataset
dim(train_set)

## types of the attributes
sapply(train_set, class)

## peek at the data itself
head(train_set)

## levels of the class attribute
levels(train_set$Species)

## breakdown of the instances in each class
percentage <- prop.table(table(train_set$Species)) * 100
cbind(freq=table(train_set$Species), percentage=percentage)

## statistical summary of all attributes
summary(train_set)


##################################################################################################################
### VISUALIZING data ###

## Univariate plots to better understand each attribute
x <- train_set[, 1:4] #predictor/feature/input/independent
y <- train_set[, 5] #target/outcome/output/dependent

# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
        boxplot(x[,i], main=names(iris)[i])
}

plot(y)

## Multivariate plots to better understand the relationships between attributes

# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")


# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)



##################################################################################################################
### RUNNING ALGORITHMS ###


# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"


# a) linear algorithms
set.seed(42)
fit.lda <- train(Species~., data=train_set, method="lda", metric=metric, trControl=control)


# b) nonlinear algorithms
# CART
set.seed(42)
fit.cart <- train(Species~., data=train_set, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(42)
fit.knn <- train(Species~., data=train_set, method="knn", metric=metric, trControl=control)


# c) advanced algorithms
# SVM
set.seed(42)
fit.svm <- train(Species~., data=train_set, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(42)
fit.rf <- train(Species~., data=train_set, method="rf", metric=metric, trControl=control)


##################################################################################################################
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# visually compare results
dotplot(results)

# select best algorithms
print(fit.knn)
print(fit.lda)


##################################################################################################################
### RUN PREDICTIONS ###

# estimate skill of LDA on the validation dataset
prediction_lda <- predict(fit.lda, test_set)
prediction_knn <- predict(fit.knn, test_set)
confusionMatrix(prediction_lda, test_set$Species)
confusionMatrix(prediction_knn, test_set$Species)
