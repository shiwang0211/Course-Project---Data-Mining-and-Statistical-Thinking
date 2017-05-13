library(rpart)
library(Metrics)
library(randomForest)
library(rpart.plot)
library(FNN)
library(kknn)
library(pls)
library(Metrics)
library(MASS)
library(kernlab)
library(neuralnet)
library(MASS)
library(nnet)
library(e1071)

### Define function used in analysis

# R-sqaure
r2 <- function(pred.y, true.y)
{ 1 - length(true.y)*mse(pred.y, true.y)/((length(true.y)-1)*var(true.y)) }

# complexity parameter corresponding to the minimum cross-validated error
cpmin <- function(cptab) { cptab[which.min(cptab[,4])[1], 1] }

# complexity parameter corresponding to the smallest tree within 1-SD from the minimum cross-validated error
cp1sd <- function(cptab)
{ cptab[which(cptab[,4]<min(cptab[,4]) + cptab[which.min(cptab[,4]),5])[1], 1] }

# create a piecewise linear model represented by an rpart regression tree
lmrpart <- function(formula, data, skip.attr=FALSE, ...)
{
m.tree <- rpart(formula, data, ...)
m.leaves <- sort(unique(predict(m.tree, data)))
m.lm <- 'names<-'(lapply(m.leaves, function(l)
lm(G3~.,
data[predict(m.tree, data)==l,])),
m.leaves)
'class<-'(list(tree=m.tree, lm=m.lm), "lmrpart")
}

# prediction method for lmrpart
predict.lmrpart <- function(model, data)
{
leaves <- as.character(predict(model$tree, data))
sapply(1:nrow(data),
function(i) predict(model$lm[[leaves[i]]], data[i,]))
}

### Read in raw data

data <- read.table("student-mat.csv",sep=";",header=TRUE)

### Generate dummy variables for discrete atribute with more than two categories

attach(data)
Mjob.f = factor(Mjob)
Mjob.dum = model.matrix(~Mjob.f)[,2:5]

Fjob.f = factor(Fjob)
Fjob.dum = model.matrix(~Fjob.f)[,2:5]

guardian.f = factor(guardian)
guardian.dum = model.matrix(~guardian.f)[,2:3]

reason.f = factor(reason)
reason.dum = model.matrix(~reason.f)[,2:4]

### Bind dummay variables with dataset

detach(data)
data<-subset(data,select = -c(Mjob, Fjob, guardian, reason))
data<-cbind(data,Mjob.dum,Fjob.dum,guardian.dum , reason.dum )

data=data
for(i in 1:ncol(data)){
data[,i]<-as.numeric(data[,i])}

### Histograms

png(filename="Histogram of G3",width = 1000, height = 1000,res=150)
hist(data[["G3"]],breaks = 20, main = "Histogram of G3", xlab = "score")
dev.off()

### Barplot
png(filename="Barplot.png",width = 1000, height = 3000,res=150)
opar <- par(mfrow = c(12, 3), oma = c(0, 0, 1.1, 0),mar = c(4.1, 4.1, 2.1, 1.1))
for(i in 1:29){
barplot(table(data[,i]), main = colnames(data)[i], xlab = "value", col="grey",horiz = TRUE, las=1)}
for(i in 30:30){
hist(data[,i],breaks = 10, main = colnames(data)[i], xlab = "value", xlim=c(0,80))}
for(i in 31:32){
hist(data[,i],breaks = 10, main = colnames(data)[i], xlab = "value", xlim=c(0,20))}
par(opar)
dev.off()

png(filename="Correlation_G1_G3.png",width = 1000, height = 1000,res=150)
plot(data$G1, data$G3, type="p",pch=21,main="G1 and G3", xlab="G1", ylab="G3")
dev.off()
cor(data$G1, data$G3)

png(filename="Correlation_G2_G3.png",width = 1000, height = 1000,res=150)
plot(data$G2, data$G3, type="p",pch=21, main="G2 and G3",xlab="G2", ylab="G3")
dev.off()
cor(data$G2, data$G3)

### data partition 3:7

L<-nrow(data)
set.seed(12345)
L_train = 0.7 * L;
label<-sample(c(1:L),L_train)

### train and test split

train<-subset(data, rownames(data) %in% label)
test<-subset(data, !(rownames(data) %in% label))

### Run by different cases

case = 2; #Include G1 and others 

if (case == 1){
ex <- names(train) %in% c("pass")
ex2 <- names(train) %in% c("G3")}

if (case == 2){
ex <- names(train) %in% c("G2")
ex2 <- names(train) %in% c("G2","G3")}

if (case == 3){
ex <- names(train) %in% c("G1","G2")
ex2 <- names(train) %in% c("G1","G2","G3")}

### LS

mod.ls <- lm(G3~., data = train[!ex])
p.ls <- predict(mod.ls,newdata = test)
r2.ls =r2 (p.ls , test$G3)
r2.ls

### Stepwise Regression 

mod.ls.step <- stepAIC(mod.ls, direction="both")
p.ls.step <- predict(mod.ls.step,newdata = test)
B.ls.step <- mod.ls.step[["coefficients"]]
r2.ls.step = r2(predict(mod.ls.step, test), test$G3)
r2.ls.step

### PCR

mod.pcr <- pcr(G3~., validation="CV", data = train[!ex])
ncomp <- which.min(mod.pcr$validation$adj)
p.pcr <- predict(mod.pcr,newdata=test,ncomp=ncomp)
B.pcr <- as.numeric(coef(mod.pcr, ncomp = ncomp, intercept = TRUE))
r2.pcr =r2 (p.pcr , test$G3)
r2.pcr

### PLS

mod.plsr <- plsr(G3~., validation="CV", data = train[!ex])
ncomp <- which.min(mod.plsr$validation$adj) #  bias-corrected mean squared error of prediction (MSEP)
p.plsr <- predict(mod.plsr,test,ncomp=ncomp)
B.plsr <- as.numeric(coef(mod.plsr, ncomp = ncomp, intercept = TRUE))
r2.plsr =r2 (p.plsr , test$G3)
r2.plsr

### Ridge

m.ridge <- lm.ridge(G3 ~ .,data=train[!ex], lambda = seq(0,50,1))
plot(m.ridge$GCV)
p.ridge = scale(test[!ex2],center = T, scale = m.ridge$scales)%*% m.ridge$coef[,which.min(m.ridge$GCV)] + m.ridge$ym
B.ridge <- as.numeric(coef(lm.ridge(G3 ~ .,data=train[!ex], lambda = m.ridge$lambda[which.min(m.ridge$GCV)])))#Best lambda
r2.ridge =r2 (p.ridge , test$G3)
r2.ridge

### Plot for regression

png(filename="Four regression.png",width = 1000, height = 1000,res=150)
opar <- par(mfrow = c(2, 2), oma = c(0, 0, 1.1, 0),mar = c(4.1, 4.1, 2.1, 1.1))
plot(p.ls.step, test$G3,pch=3,main="stepwise",ylab="Actual G3",xlab="Predicted G3",xlim=c(0,20),ylim=c(0,20))
plot(p.pcr, test$G3,pch=3,main="PCR",ylab="Actual G3",xlab="Predicted G3",xlim=c(0,20),ylim=c(0,20))
plot(p.plsr, test$G3,pch=3,main="PLSR",ylab="Actual G3",xlab="Predicted G3",xlim=c(0,20),ylim=c(0,20))
plot(p.ridge, test$G3,pch=3,main="Ridge",ylab="Actual G3",xlab="Predicted G3",xlim=c(0,20),ylim=c(0,20))
dev.off()

### Decision Tree

control <- rpart.control(minsplit = 2,minbucket = 10,cp=0)
fit.dt<-rpart(G3 ~., data = train[!ex],control = control,method="anova")

fit.dt.pmin = prune(fit.dt, cpmin(fit.dt$cptable))  
fit.dt.p1sd = prune(fit.dt, cp1sd(fit.dt$cptable))

r2.dt.pmin = r2(predict(fit.dt.pmin, test, type="vector"),test[["G3"]])     
r2.dt.pmin
r2.dt.p1sd = r2(predict(fit.dt.p1sd, test, type="vector"),test[["G3"]])
r2.dt.p1sd

png(filename="Unpruned decision tree",width = 1000, height = 1000,res=150)
prp(fit.dt)
dev.off()

png(filename="Pruned decision tree",width = 1000, height = 1000,res=150)
prp(fit.dt.p1sd)
dev.off()

### Random forest

fit.rf <- randomForest(G3 ~., data=train[!ex], importance=TRUE)

p.rf<-predict(fit.rf, test, type="response")
r2.rf = r2(p.rf,test[["G3"]])
r2.rf

attr.utl<-sort(importance(fit.rf)[,1],decreasing = TRUE)
asets<-
'names<-'(lapply(c(10,25,50,100),
function(p)
names(attr.utl)[1:round(p*length(attr.utl)/100)]),
paste(c(10,25,50,100),"percent",sep=" "))

png(filename="Variable importance",width = 1000, height = 1000,res=150)
varImpPlot(fit.rf, type=1) #PLOT IMPORTANCE
dev.off()

### Decision tree with variable selection

fit.dt.group <-
lapply(asets,
function(aset)
{

tree.full <- rpart(make.formula("G3", aset), train[!ex],
minsplit=2, minbucket=10, cp=0)
tree.pmin <- prune(tree.full, cpmin(tree.full$cptable))
tree.p1sd <- prune(tree.full, cp1sd(tree.full$cptable))
list(
tree.pmin=tree.pmin,
r2.pmin=r2(predict(tree.pmin, test), test$G3),
tree.p1sd=tree.p1sd,
r2.p1sd=r2(predict(tree.p1sd, test), test$G3))
})

write.table(
sapply(fit.dt.group,
function(tree) c(r2.pmin=tree$r2.pmin, r2.p1sd=tree$r2.p1sd)),
file=paste("case ",case," Decision tree assets"))

png(filename="Decision-tree-best-subset",width = 1000, height = 1000,res=150)
prp(fit.dt.group$`10 percent`$tree.p1sd, varlen=0, faclen=0)
dev.off()

### Piece-wise linear regression

control <- rpart.control(minbucket = 10,cp=0.1,maxdepth=7,xval = 10)
lmtree<-lmrpart(make.formula("G3", asets$"10 percent"), train[,c("G3",asets$"10 percent")], control = control)
p.lmtree= predict.lmrpart(lmtree, test[,asets$"10 percent"])

r2.lmtree = r2(p.lmtree,test[["G3"]])
r2.lmtree

png(filename="LMTREE",width = 1000, height = 1000,res=150)
prp(lmtree$tree, varlen=0, faclen=0)
dev.off()

### kNN
for(x.sel in 0:1){
if(x.sel == 0) {train.knn = train[!ex2]; test.knn = test[!ex2]}
if(x.sel == 1) {train.knn = train[,c(asets$"10 percent")]; test.knn = test[,c(asets$"10 percent")]}

cv.err.knn = sapply(1:30,
function(k) knn.reg(train.knn,y=train[["G3"]],k=k)$PRESS)

k.best=which.min(cv.err.knn)

png(filename="kNN-selection_of_k",width = 1000, height = 1000,res=150)
plot(c(1:30),cv.err.knn,type="l",ylab="sum of squared residual", xlab="value of k", main="Leave-One-Out CV for kNN")
dev.off()

p.knn<-knn.reg(train.knn, test.knn, train[["G3"]], k=k.best,algorithm=c("kd_tree"))$pred
if (x.sel == 0) r2.knn = r2(p.knn,test[["G3"]])
if (x.sel == 1) r2.knn.xsel = r2(p.knn,test[["G3"]])
}

### NN

for(x.sel in 0:1){
if(x.sel == 0) {train.nn = train[!ex]; test.nn = test[!ex2]}
if(x.sel == 1) {train.nn = train[,c("G3",asets$"10 percent")]; test.nn = test[,c(asets$"10 percent")]}

fit.net<-best.nnet(G3/20~., data = train.nn, size = 1:10,maxit = 1000,tunecontrol=tune.control(cross=10))
p.net<-predict(fit.net,test.nn, type="raw")*20

if (x.sel == 0) r2.net<-r2 (p.net, test$G3)
if (x.sel == 1) r2.net.xsel <- r2 (p.net, test$G3)
}

### SVM
for(x.sel in 0:1){

if(x.sel == 0) train.svm = train[!ex];
if(x.sel == 1) train.svm = train[,c("G3",asets$"10 percent")];

cv.err.svm = sapply(1:20,
function(C) cross(ksvm(G3~.,data=train.svm,kernel="rbfdot",type="nu-svr",cross=10,C=C)))

C.best = which.min(cv.err.svm)
fit.svm<-svm(G3~.,data=train.svm,kernel="radial",cost=C.best)

p.svm<-predict(fit.svm,newdata=test,type="response")
if (x.sel == 0) r2.svm<-r2(p.svm, test$G3)
if (x.sel == 1) r2.svm.xsel<-r2(p.svm, test$G3)
}

### Report error rate
X = c(r2.ls.step,r2.pcr,r2.plsr,r2.ridge,r2.dt.pmin,r2.dt.p1sd,r2.rf,r2.lmtree,r2.net,r2.net.xsel,r2.svm,r2.svm.xsel,r2.knn,r2.knn.xsel)
write.table(X,file=paste("case ",case," Test-error"),row.names=FALSE, col.names=FALSE)
write.table(cbind(c("Intercept",colnames(train[!ex2])) ,B.pcr,B.plsr,B.ridge), file="Coefficients",row.names=FALSE, col.names=FALSE)
