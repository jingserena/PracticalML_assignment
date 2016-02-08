
traindata <- read.csv("C:\\Users\\JHuo\\Desktop\\coursera\\machine Learning\\pml-training.csv", as.is=TRUE)
testdata <- read.csv("C:\\Users\\JHuo\\Desktop\\coursera\\machine Learning\\pml-testing.csv", as.is=TRUE)

# remove the first 7 identifier columns, empty columns and those with lots of NAs, by visual observation
traindata <- traindata[, c(8:11, 37:49, 60:68, 84:86, 102,113:124, 140, 151:ncol(traindata) )]
testdata <- testdata[, c(8:11, 37:49, 60:68, 84:86, 102,113:124, 140, 151:ncol(testdata) )]

# remove those that have few variance
library(caret)
#nsv <- nearZeroVar(traindata, saveMetrics=TRUE)
nsv <- nearZeroVar(traindata)
traindata_nsv <- traindata
if(length(nsv) > 0) traindata_nsv <- traindata[, -nsv]

# split the traindata into training and validation sets
inTrain <- createDataPartition(y=traindata_nsv$classe, p=0.7, list=FALSE)
training <- traindata[inTrain,]
testing <- traindata[-inTrain,]

# remove those rows with NAs
cc <- complete.cases(training)
training <- training[cc,]

# rescale into n(0,1)
library(kernlab)
preObj <- preProcess(training[,-53], method=c("center","scale"))
train_scale <- predict(preObj, training[,-53])
test_scale <- predict(preObj, testing[,-53])

# see correlation
M <- abs(cor(train_scale))
diag(M)<-0
which(M>0.8, arr.ind=T)

plot(train_scale$accel_belt_x, train_scale$pitch_belt)

par(mfrow=c(1,3))
plot(train_scale[,31], train_scale[,46])
plot(train_scale[,33], train_scale[,46])
plot(train_scale[,45], train_scale[,46])

# run modeling
library(randomForest)
clf_200 <- randomForest(factor(training$classe)~., data=train_scale, ntree=200, nodesize=5)
varImpPlot(clf_200, sort=TRUE, main="variable importance from random forest")

resProb <- predict(clf_200, test_scale, type="response")
xtab=table(resProb, testing$classe)
print(confusionMatrix(xtab))

test_test_scale <- predict(preObj, testdata[,-53])
resProb <- predict(clf_200, test_test_scale, type="response")
xtab=table(resProb, testdata$classe)
print(confusionMatrix(xtab))
