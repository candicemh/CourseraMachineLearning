#Coursera Project Write-up

## Background to the Project

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Data for the Project

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

## Goals of the Project 

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. Other variables are used to predict "classe". This report describes how I built the prediction model, how cross validation was used, what the expected out of sample error is, and why I made the choices I did. The prediction model is used to predict 20 different test cases. 

## R code for prediction

Set working directory and load libraries. Read in the data. Then, delete empty or almost empty columns, and identifying columns that cannot explain the response (such as time).
<pre><code>
setwd("/Users/Candice/Desktop")

library(caret)
library(kernlab)
library(rpart)

filename_test<-"pml-testing.csv"
filename_train<-"pml-training.csv"
url_test<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-test.csv"
url_training<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
if (!file.exists(filename_test)) {
	download.file(url_test,filename_test)
}
if (!file.exists(filename_train)) {
	download.file(url_train,filename_train)
}
data_test<-read.csv(filename_test, header=TRUE,sep=",")
data_train<-read.csv(filename_train, header=TRUE,sep=",")

throwcols<-grep("avg|stddev|var|min|max|amplitude|skewness|kurtosis|timestamp|user_name|new_window",names(data_test),value=FALSE)
data_test<-data_test[,-throwcols]
throwcols<-grep("avg|stddev|var|min|max|amplitude|skewness|kurtosis|timestamp|user_name|new_window",names(data_train),value=FALSE)
data_train<-data_train[,-throwcols]

data_test[,1]<-NULL
data_train[,1]<-NULL
</code></pre>


Now that we are working with clean data, subset the training into train and test, 75% of the training data is used to build the model, and 25% is used to  validate the model, and to estimate the out of sample error
<pre><code>
inTrain<-createDataPartition(y=data_train$classe,p=0.75,list=FALSE)
subset_train<-data_train[inTrain,]
subset_test<-data_train[-inTrain,]
</code></pre>

PCA is for dimension reduction - chosen because 53 explanatory variables is too large to be practical, and it is obvious from the names of the variables that we expect them to be related and highly correlated. The scree plot from the PCA shows us that the first 10 components explains 95% of the information in the original training data subset, and the principle of parsimony suggests that this is sufficient for prediction purposes to prevent overfitting.
<pre><code>
preProc_train<-preProcess(subset_train[,-54],method="pca",pcaComp=10)
trainpred<-predict(preProc_train,subset_train[,-54])
testpred<-predict(preProc_train,subset_test[,-54])
</code></pre>

Since the outcome is categorical and the explanatory variables are all continuous random variables, the prediction model is built using Random Forests using the subsetting training data. The model is then applied to the subsetted testing data to estimate the confusion matrix and consequently the expected out of sample error rate.
<pre><code>
modelfit_rf1<-train(x=trainpred,y=subset_train$classe,method="rf")
pred_rf1<-predict(modelfit_rf1, testpred)
confusionMatrix(pred_rf1,subset_test$classe)
</code></pre>

>Confusion Matrix and Statistics
>
>          Reference
>Prediction    A    B    C    D    E
>         A 1362   25    8    8    6
>         B    8  891   18    1    2
>         C   13   20  813   34    3
>         D    8    7   12  758   15
>         E    4    6    4    3  875
>
>Overall Statistics
>                                          
>               Accuracy : 0.9582          
>                 95% CI : (0.9522, 0.9636)
>    No Information Rate : 0.2845          
>    P-Value [Acc > NIR] : < 2.2e-16       
>                                          
>                  Kappa : 0.9471          
> Mcnemar's Test P-Value : 9.79e-05        
>
>Statistics by Class:
>
>                     Class: A Class: B Class: C Class: D Class: E
>Sensitivity            0.9763   0.9389   0.9509   0.9428   0.9711
>Specificity            0.9866   0.9927   0.9827   0.9898   0.9958
>Pos Pred Value         0.9666   0.9685   0.9207   0.9475   0.9809
>Neg Pred Value         0.9906   0.9854   0.9896   0.9888   0.9935
>Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
>Detection Rate         0.2777   0.1817   0.1658   0.1546   0.1784
>Detection Prevalence   0.2873   0.1876   0.1801   0.1631   0.1819
>Balanced Accuracy      0.9815   0.9658   0.9668   0.9663   0.9834

This implies that the out of sample error will be approximately 95.82%. We now apply the prediction model to the project dataset to predict the outcomes.
<pre><code>
projectpred<-predict(preProc_train,data_test[,-54])
project<-predict(modelfit_rf1, projectpred)
</code></pre>

Project now contains the 20 predictions, which in order are: 

>B A A A A E D B A A B C B A E E A B B B. 

The remainder of the code simply writes these predictions to text files for uploading for assessment.
<pre><code>
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(project)
</code></pre>
