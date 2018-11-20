typos = c(2,3,0,3,1,0,0,1)
typos

mean(typos)
median(typos)
var(typos)

typos.draft1 = c(2,3,0,3,1,0,0,1)
typos.draft2 = c(0,3,0,3,1,0,0,1)

typos.draft1 = c(2,3,0,3,1,0,0,1)
typos.draft2 = typos.draft1   # make a copy
typos.draft2[1] = 0  # assign the first page 0 typos
typos.draft2  # print out the value
typos.draft2[2]  # print 2nd pages' value
typos.draft2[4]  # 4th page
typos.draft2[-4] # all but the 4th page
typos.draft2[c(1,2,3)]  # fancy, print 1st, 2nd, 3rd

max(typos.draft2)
typos.draft2 == 3

which(typos.draft2 == 3)

n = length(typos.draft2)  # how many pages
pages = 1:n  #how we get the page numbers
pages  
pages[typos.draft2 == 3]  #logical extraction. very useful


(1: length(typos.draft2))[typos.draft2 == max(typos.draft2)]

sum(typos.draft2) #how many typos?
sum(typos.draft2 >0) #how many pages with typos?
typos.draft1 - typos.draft2  #difference between the two



#example: keeping track of a stock; adding to the data
x = c(45,43,46,48,51,46,50,47,46,45)
mean(x)
median(x)
max(x)
min(x)

x = c(x,48,49,51,50,49) #append values to x 
length(x) #how long is x now (it was 10)
x[16] = 41 # add to a specified index
x[17:20] = c(40,38,35,40) #add to many specified indeces



#page7

data.entry(x) # Pops up spreadsheet to edit data
x = de(x) # same only, doesn¡¯t save changes
x = edit(x) # uses editor to edit x


data.entry(x)
data.entry(x=c(NA)) # works, x is defined as we go

day = 5;
mean(x[day:(day+4)])
day:(day+4) 

cummax(x)  # running maximum
cummin(x)  # running minimum


#page 8

whale = c(74, 122, 235, 111, 292, 111, 211, 133, 156, 79) 
mean(whale) 
var(whale)
std(whale)
sqrt(var(whale))
sqrt(sum ((whale - mean(whale)) ^2 / (length(whale)-1)))

std = function(x) sqrt((var(x)))
std(whale)

sd(whale)


#page11

#example: smking survey

x=c("Yes","No","No","Yes","Yes") 
table(x)

x=c("Yes","No","No","Yes","Yes") 
x
factor(x) # print out value in factor(x) 

beer = scan() 
#####

beer.counts = table(beer) # store the table result
pie(beer.counts)  # first pie -- kind of dull 
names(beer.counts) =c("domestic\n can", "Domestic\n bottle", "Microbrew","Import")  # give names 
pie(beer.counts)
pie(beer.counts, col = c("purple","green2","cyan","white")) # now with colors


#page 14
sals = scan() 

data=c(10, 17, 18, 25, 28, 28)
summary(data)
quantile(data,.25)
quantile(data,c(.25,.75)) 

#page16
sort(sals)
fivenum(sals) 
summary(sals) 
mean(sals,trim=1/10) 
mean(sals,trim=2/10) 


#page 17
IQR(sals)
mad(sals)

median(abs(sals - median(sals))) 
median(abs(sals - median(sals))) * 1.4826 

scores = scan() 


#page18
stem(scores,scale=2)

sals = c(12, .4, 5, 2, 50, 8, 3, 1, 4, .25) # enter data 
cats = cut(sals,breaks=c(0,1,5,max(sals))) # specify the breaks 
cats
table(cats) # organize 
levels(cats) = c("poor","rich","rolling in it") # change labels
table(cats)

#page19



#page21
library("UsingR")
data(movies)
names(movies)
attach(movies)  # to access the names above 
boxplot(current,main="current receipts",horizontal=TRUE)
boxplot(gross,main="gross receipts",horizontal=TRUE) 
detach(movies)  # tidy up


#page 22
library("timeSeries")
data("lynx")
summary(lynx)

#page 23
x = c(.314,.289,.282,.279,.275,.267,.266,.265,.256,.250,.249,.211,.161) 
tmp = hist(x)
lines(c(min(tmp$breaks),tmp$mids,max(tmp$breaks)),c(0,tmp$counts,0),type="l")

#page 24
data(faithful)
attach(faithful)   # make eruptions visible 
hist(eruptions,15,prob=T)   # proportions, not frequencies 
lines(density(eruptions))    # lines makes a curve, default bandwidth 
lines(density(eruptions, bw="SJ"), col='red')

#page 26
smokes = c("Y","N","N","Y","N","Y","Y","Y","N","Y")
amount = c(1,2,2,3,3,1,2,1,3,2)
table(smokes, amount)

tmp = table(smokes, amount)  # store the table 
old.digits = options("digits")  # store the number of digits 
options(digits=3)  # only print 3 decimal places
prop.table(tmp,1)  # the rows sum to 1 now 
prop.table(tmp,2)  # the columns sum to 1 now 

#page 27
prop.table(tmp)
#options(digits = old.digits)

barplot(table(smokes,amount))
barplot(table(amount, smokes))
smokes = factor(smokes)
barplot(table(smokes,amount),  beside=TRUE, legend.text=T)
barplot(table(amount,smokes),main="table(amount,smokes)",beside=TRUE, legend.text=c("less than 5","5-10","more than 10"))



#page 28
prop = function(x) x/sum(x)
apply(x,2,prop) #¿¡·¯¶ä

t(apply(x,1,prop))  #¿¡·¯¶ä


x = c(5, 5, 5, 13, 7, 11, 11, 9, 8, 9) 
y = c(11, 8, 4, 5, 9, 5, 10, 5, 4, 10) 
boxplot(x,y)

library("UsingR");data(home)
attach(home)
names(home)
boxplot(scale(old),scale(new))
detach(home)

#page30
stripchart(scale(old),scale(new))  #¿¡·¯¶ä
simple.violinplot(scale(old),scale(new))  #¿¡·¯¶ä



data(home);attach(home) 
plot(old,new)
detach(home)

#page 31
data(homedata)
attach(homedata)
plot(old,new)
detach(homedata)

x = 1:2;y = c(2,4);df = data.frame(x=x,y=y) 
ls()
rm(y)
attach(df)
ls()
ls(pos = 2)
y
x
x = c(1,3)
df
detach(df)
x
y


#page32
data(home);attach(home)
x = old  # use generic variable names 
y = new  # for illustration only
plot(x,y)
abline(lm(y~x))
detach(home)

#page 33
data(home);attach(home)
x=old; y = new
simple.lm(x,y)
detach(home)

lm.res = simple.lm(x,y)
coef(lm.res)
coef(lm.res)[1]

simple.lm(x,y,show.residuals=TRUE)

#page 34

lm.res = simple.lm(x,y)
the.residuals = resid(lm.res)
plot(the.residuals)

cor(x,y)
#page 35
cor(x,y) ^2

rank(c(2,3,5,7,11))
rank(c(5,3,2,7,11))
rank(c(5,5,2,7,5))

cor(rank(x), rank(y))
cor.sp <- function(x,y) cor(rank(x), rank(y))
cor.sp(x,y)

#page 36 ~37  #´Ù½ÃÇÏ±â
data("florida")
names(florida)
attach(florida)
simple.lm(BUSH,BUCHANAN)

detach(florida)

identify(BUSH,BUCHANAN,n=2) # ¿¡·¯¶ä
BUSH[50]
florida[50,]  

simple.lm(BUSH[-50],BUCHANAN[-50])  #¿¡·¯¶ä
65.57350 + 0.00348 * BUSH[50]
simple.lm(BUSH[-50],BUCHANAN[-50],pred=BUSH[50]) 
simple.lm(BUSH,BUCHANAN) 
abline(65.57350,0.00348)


# page 38 #¿©±âµµ ´Ù½Ã
library(MASS)
attach(florida)
plot(BUSH,BUCHANAN) 
abline(lm(BUCHANAN ~ BUSH),lty="1") 
abline(rlm(BUCHANAN ~ BUSH),lty="2") 
legend(locator(1),legend=c("lm","rlm"),lty=1:2)
detach(florida)
plot(BUSH,BUCHANAN) 
abline(rlm(BUCHANAN ~ BUSH),lty="1") 
abline(rlm(BUCHANAN[-50] ~ BUSH[-50]),lty="2")

#page 39 ~ 40
x=seq(0,4,by=.1)
plot(x,x^2,type="l") # type="l" to make line
curve(x^2,0,4)
miles = (0:8)*4 
tread = scan() 
plot(miles,tread) 
abline(lm(tread ~ miles))
abline(360,-7.3) 
points(miles,360 - 7.3*miles,type="l") 
lines(miles,360 - 7.3*miles) 
curve(360 - 7.3*x,add=T)

# page 42 ~ 44
weight = c(150, 135, 210, 140) 
height = c(65, 61, 70, 65) 
gender = c("Fe","Fe","M","Fe") 
study = data.frame(weight,height,gender)   # make the data frame 
study 

study = data.frame(w=weight,h=height,g=gender)
row.names(study)<-c("Mary","Alice","Bob","Judy")
study
rm(weight)
weight
attach(study) 
w
#study[,"weight"]
study[,1]
study[,1:2]
study["Mary",] 
study["Mary","w"] 
study$w   # using $
study[["weight"]] 
study[["w"]] # unambiguous shortcuts are okay
study[[1]]  # by position
study[study$g == "Fe", ] # use $ to access gender via a list 

#page 45
data(PlantGrowth) 
PlantGrowth 
attach(PlantGrowth) 
weight.ctrl = weight[group == "ctrl"]
unstack(PlantGrowth) 
boxplot(unstack(PlantGrowth))
boxplot(weight ~ group)

#page 46 ~47
library(MASS);data(Cars93);attach(Cars93) 
price = cut(Price,c(0,12,20,max(Price)))
levels(price)=c("cheap","okay","expensive")
mpg = cut(MPG.highway,c(0,20,30,max(MPG.highway))) 
levels(mpg) = c("gas guzzler","okay","miser")
table(Type) 
table(price,Type) 
table(price,Type,mpg) 
barplot(table(price,Type),beside=T) # the price by different types
barplot(table(Type,price),beside=T) # type by different prices

y=rnorm(1000) 
f=factor(rep(1:10,100)) 
boxplot(y ~ f,main="Boxplot of normal random data with model notation")

#page 48
x = rnorm(100) 
y = factor(rep(1:10,10)) 
stripchart(x ~ y)

par(mfrow=c(1,3)) 
data(InsectSprays) 
boxplot(count ~ spray, data = InsectSprays, col = "lightgray") 
simple.violinplot(count ~ spray, data = InsectSprays, col = "lightgray") #¿¡·¯¶ä
simple.densityplot(count ~ spray, data = InsectSprays) #¿¡·¯¶ä

#page 49
plot(x,y)
points(x,y,pch="2")  

data("ToothGrowth") 
attach(ToothGrowth) 
plot(len ~ dose,pch=as.numeric(supp)) 
tmp = levels(supp) 
legend(locator(1),legend=tmp,pch=1:length(tmp)) 
detach(ToothGrowth)


#page 50
#data(emissions)
#attach(emissions) 
#simple.scatterplot(perCapita,CO2) 
#title("GDP/capita vs. CO2 emissions 1999") 
#detach(emissions)
#pairs(emissions)


#page 52
histogram( ~ Max.Price | Cylinders , data = Cars93)
bwplot( ~ Max.Price | Cylinders , data = Cars93)

attach(Cars93) 
xyplot(MPG.highway ~ Fuel.tank.capacity | Type) 
plot.regression = function(x,y) { panel.xyplot(x,y) + panel.abline(lm(y~x))  } 
trellis.device(bg="white") 
xyplot(MPG.highway ~ Fuel.tank.capacity | Type, panel = plot.regression)


#page 54 ~55
sample(1:6,10,replace=T) 
RollDie = function(n) sample(1:6,n,replace=T) 
RollDie(5)

runif(1,0,2) 
runif(5,0,2) 
runif(5)
x=runif(100) 
hist(x,probability=TRUE,col=gray(.9),main="uniform on [0,1]") 
curve(dunif(x,0,1),add=T)

rnorm(1,100,16) 
rnorm(1,mean=280,sd=10) 
x=rnorm(100) 
hist(x,probability=TRUE,col=gray(.9),main="normal mu=0,sigma=1") 
curve(dnorm(x),add=T) 

#page 56
n=1 
p=.5 
rbinom(1,n,p) 
rbinom(10,n,p) 

n = 10; p=.5 
rbinom(1,n,p) # 6 successes in 10 trials
rbinom(5,n,p)  # 5 binomial number 

#page 57
n=5;p=.25 
x=rbinom(100,n,p)
hist(x,probability=TRUE,)
xvals=0:n;points(xvals,dbinom(xvals,n,p),type="h",lwd=3) 
points(xvals,dbinom(xvals,n,p),type="p",lwd=3)


x=rexp(100,1/2500) 
hist(x,probability=TRUE,col=gray(.9),main="exponential mean=2500") 
curve(dexp(x,1/2500),add=T)


#page 58
sample(1:6,10,replace=TRUE)
sample(c("H","T"),10,replace=TRUE)
sample(1:54,6) 
cards = paste(rep(c("A",2:10,"J","Q","K"),4),c("H","D","S","C"))
sample(cards,5) 
dice = as.vector(outer(1:6,1:6,paste)) 
sample(dice,5,replace=TRUE)

data(faithful) 
names(faithful) 
eruptions = faithful[["eruptions"]] 
sample(eruptions,10,replace=TRUE) 
hist(eruptions,breaks=25) 
hist(sample(eruptions,100,replace=TRUE),breaks=25)


#page 60 ~64
pnorm(.7) 
pnorm(.7,1,1) 
pnorm(.7,lower.tail=F) 
qnorm(.75) 

x = rnorm(5,100,16) 
x 
z = (x-100)/16 
z
pnorm(z) 
pnorm(x,100,16) 

n=10;p=.25;S= rbinom(1,n,p) 
(S - n*p)/sqrt(n*p*(1-p)) 
n = 10;p = .25;S = rbinom(100,n,p) 
X = (S - n*p)/sqrt(n*p*(1-p))
hist(X,prob=T)
results =numeric(0) 
#for (i in 1:100) { S = rbinom(1,n,p)  results[i]=(S- n*p)/sqrt(n*p*(1-p)) }
 
primes=c(2,3,5,7,11); 
for(i in 1:5) print(primes[i]) 
for(i in primes) print(i)
results = c(); 
mu = 0; sigma = 1 

#page 65 ~67for¹® ´Ù½ÃÇÏ±â
x = rnorm(100,0,1);qqnorm(x,main="normal(0,1)");qqline(x)
x = rnorm(100,10,15);qqnorm(x,main="normal(10,15)");qqline(x)
x = rexp(100,1/10);qqnorm(x,main="exponential mu=10");qqline(x) 
x = runif(100,0,1);qqnorm(x,main="unif(0,1)");qqline(x)

#page 67
f = function(n=100,mu=10) (mean(rexp(n,1/mu))-mu)/(mu/sqrt(n))
xvals = seq(-3,3,.01)
hist(simple.sim(100,f,1,10),probability=TRUE,main="n=1",col=gray(.95))
points(xvals,dnorm(xvals,0,1),type="l") 

#page 71 ~73
data(homedata) 
attach(homedata) 
hist(y1970);hist(y2000) 
detach(homedata) 

attach(homedata) 
simple.eda(y1970);simple.eda(y2000) 
detach(homedata) 

data(exec.pay) 
simple.eda(exec.pay)
log.exec.pay = log(exec.pay[exec.pay >0])/log(10)
simple.eda(log.exec.pay)


data(ewr) 
names(ewr)
airnames = names(ewr) 
ewr.actual = ewr[,3:10] 
boxplot(ewr.actual)

par(mfrow=c(2,4)) 
attach(ewr) 
for(i in 3:10) boxplot(ewr[,i]~ as.factor(inorout),main=airnames[i]) 
detach(ewr)
par(mfrow=c(1,1)) 


#page 75
X=runif(100);boxplot(X,horizontal=T,bty=n) 
X=rnorm(100);boxplot(X,horizontal=T,bty=n) 
X=rt(100,2);boxplot(X,horizontal=T,bty=n)
X=sample(1:6,100,p=7-(1:6),replace=T);boxplot(X,horizontal=T,bty=n) 
X=abs(rnorm(200));boxplot(X,horizontal=T,bty=n) 
X=rexp(200);boxplot(X,horizontal=T,bty=n)


#page 78 ~79
alpha = c(0.2,0.1,0.05,0.001) 
zstar = qnorm(1 - alpha/2) 
zstar
2*(1-pnorm(zstar)) 

m = 50; n=20; p = .5; 
phat = rbinom(m,n,p)/n
SE = sqrt(phat*(1-phat)/n) 
alpha = 0.10;zstar = qnorm(1-alpha/2) 
matplot(rbind(phat - zstar*SE, phat + zstar*SE), + rbind(1:m,1:m),type="l",lty=1) > abline(v=p) # draw line for p=0.5

#page 80 ~81
prop.test(42,100)
prop.test(42,100,conf.level=0.90)

#simple.z.test = function(x,sigma,conf.level=0.95) { + n = length(x);xbar=mean(x) + alpha = 1 - conf.level + zstar = qnorm(1-alpha/2) + SE = sigma/sqrt(n) + xbar + c(-zstar*SE,zstar*SE) + 
simple.z.test(x,1.5) 

 
#page 82 ~ 83
t.test(x)
x=rnorm(100);y=rt(100,9) 
boxplot(x,y) 
qqnorm(x);qqline(x) 
qqnorm(y);qqline(y)

xvals=seq(-4,4,.01) 
plot(xvals,dnorm(xvals),type="l") 
for(i in c(2,5,10,20,50)) points(xvals,dt(xvals,df=i),type="l",lty=i)

#page 84
x = c(110, 12, 2.5, 98, 1017, 540, 54, 4.3, 150, 432)
wilcox.test(x,conf.int=TRUE)

#page 86 ~88
prop.test(42,100,p=.5)
prop.test(420,1000,p=.5)
xbar=22;s=1.5;n=10
t = (xbar-25)/(s/sqrt(n)) 
t
pt(t,df=n-1) 

x = c(12.8,3.5,2.9,9.4,8.7,.7,.2,2.8,1.9,2.8,3.1,15.8) 
stem(x) 

wilcox.test(x,mu=5,alt="greater")
x = c(12.8,3.5,2.9,9.4,8.7,.7,.2,2.8,1.9,2.8,3.1,15.8) 
simple.median.test(x,median=5) 
simple.median.test(x,median=10) 

#page 89 ~ 93
prop.test(c(45,56),c(45+35,56+47)) 

x = c(15, 10, 13, 7, 9, 8, 21, 9, 14, 8) 
y = c(15, 14, 12, 8, 14, 7, 16, 10, 15, 12) 
t.test(x,y,alt="less",var.equal=TRUE)

t.test(x,y,alt="less")

x = c(3, 0, 5, 2, 5, 5, 5, 4, 4, 5) 
y = c(2, 1, 4, 1, 4, 3, 3, 2, 3, 5) 
t.test(x,y,paired=TRUE) 
t.test(x,y) 

data(ewr) 
attach(ewr) 
tmp=subset(ewr, inorout == "out",select=c("AA","NW")) 
x=tmp[["AA"]] 
y=tmp[["NW"]] 
boxplot(x,y)
wilcox.test(x,y) 

#page 94
x = rchisq(100,5);y=rchisq(100,50) 
simple.eda(x);simple.eda(y)

#page 95 ~97
freq = c(22,21,22,27,22,36) 
probs = c(1,1,1,1,1,1)/6 # or use rep(1/6,6)
chisq.test(freq,p=probs)

x = c(100,110,80,55,14) 
probs = c(29, 21, 17, 17, 16)/100 
chisq.test(x,p=probs)

yesbelt = c(12813,647,359,42)
nobelt = c(65963,4000,2642,303) 
chisq.test(data.frame(yesbelt,nobelt))

#page 98
die.fair = sample(1:6,200,p=c(1,1,1,1,1,1)/6,replace=T) 
die.bias = sample(1:6,100,p=c(.5,.5,1,1,1,2)/6,replace=T) 
res.fair = table(die.fair);res.bias = table(die.bias) 
rbind(res.fair,res.bias) 

chisq.test(rbind(res.fair,res.bias)) 
chisq.test(rbind(res.fair,res.bias))[["exp"]] 
#-------------------------------------------------------
#page 101 ~102
x = c(18,23,25,35,65,54,34,56,72,19,23,42,18,39,37)
y = c(202,186,187,180,156,169,174,172,153,199,193,174,198,183,178)
plot(x,y) 
abline(lm(y ~ x)) 
lm(y ~ x) 

lm.result=simple.lm(x,y) 
summary(lm.result)

coef(lm.result) 
lm.res = resid(lm.result) 
summary(lm.res) 


plot(lm.result)



#page 104 ~105
es = resid(lm.result) 
b1 =(coef(lm.result))[["x"]] 
s = sqrt( sum( es^2 ) / (n-2) )
SE = s/sqrt(sum((x-mean(x))^2)) 
t = (b1 - (-1) )/SE 
pt(t,13,lower.tail=FALSE) 

SE = s * sqrt( sum(x^2)/( n*sum((x-mean(x))^2)))
b0 = 210.04846 
t = (b0 - 220)/SE 
pt(t,13,lower.tail=TRUE)

#page 106 ~107
simple.lm(x,y,show.ci=TRUE,conf.level=0.90)
lm.result = lm(y ~ x)
summary(lm.result) 

plot(x,y) 
abline(lm.result)

resid(lm.result) 
coef(lm.result) 
coef(lm.result)[1] 
coef(lm.result)["x"] 
fitted(lm.result) 
coefficients(lm.result) 
coefficients(summary(lm.result)) 
coefficients(summary(lm.result))[2,2] 
coefficients(summary(lm.result))["x","Std. Error"] 


#----------------------start !!!!! ----
library("UsingR")
#page 108

predict(lm.result,data.frame(x= c(50,60)))
#predict(lm.result,data.frame(x=sort(x)), + level=.9, interval="confidence")
plot(x,y) 
abline(lm.result) 
#ci.lwr = predict(lm.result,data.frame(x=sort(x)), + level=.9,interval="confidence")[,2] 
points(sort(x), ci.lwr,type="l") # or use lines()
#curve(predict(lm.result,data.frame(x=x), + interval="confidence")[,3],add=T)


#page 111 ~112
x = 1:10 
y = sample(1:100,10) 
z = x+y 
lm(z ~ x+y) 

z = x+y + rnorm(10,0,2)
lm(z ~ x+y) 

z = x+y + rnorm(10,0,10) 
lm(z ~ x+y) 
lm(z ~ x+y -1) 

summary(lm(z ~ x+y )) 



#page 114 
dist = c(253, 337,395,451,495,534,574) 
height = c(100,200,300,450,600,800,1000) 
lm.2 = lm(dist ~ height + I(height^2))
lm.3 = lm(dist ~ height + I(height^2) + I(height^3)) 

lm.2
lm.3
quad.fit = 200.211950 + .706182 * pts -0.000341 * pts^2 
cube.fit = 155.5 + 1.119 * pts - .001234 * pts^2 + .000000555 * pts^3 
plot(height,dist)
lines(pts,quad.fit,lty=1,col="blue")
lines(pts,cube.fit,lty=2,col="red") 
legend(locator(1),c("quadratic fit","cubic fit"),lty=1:2,col=c("blue","red"))
summary(lm.3) 

#page 116  ¿¡·¯ #$#$#$#$#$#$#$#$#$#$#$
pts = seq(min(height),max(height),length=100) 
makecube = sapply(pts,function(x) coef(lm.3) %*% x^(0:3))
makesquare = sapply(pts,function(x) coef(lm.2) %*% x^(0:2))
lines(pts,makecube,lty=1) 
lines(pts,makesquare,lty=2)


#page 117
x = c(4,3,4,5,2,3,4,5) 
y = c(4,4,5,5,4,5,4,4) 
z = c(3,4,2,4,5,5,4,4) 
scores = data.frame(x,y,z) 
boxplot(scores)
scores = stack(scores)
names(scores) 
oneway.test(values ~ ind, data=scores, var.equal=T)


#page 120
df = stack(data.frame(x,y,z)) # prepare the data 
oneway.test(values ~ ind, data=df,var.equal=T)
anova(lm(values ~ ind, data=df)) 

kruskal.test(values ~ ind, data=df)

#page 123 ~126

data(mtcars) 
names(mtcars) 

mpg 
attach(mtcars) 
mpg 
table(cyl) 
barplot(cyl) 
barplot(table(cyl))
#stem(mpg)
#hist(mpg)
#boxplot(mpg)
#mean(mpg) 
#mean(mpg,trim=.1) 
#summary(mpg) 
#sd(mpg) 
#IQR(mpg) 
#mad(mpg) 
#mpg[cyl == 4] 
#mean(mpg[cyl == 4]) 
#plot(cyl,mpg)
#simple.lm(cyl,mpg)
#tapply(mpg,cyl,mean) 
#simple.lm(hp,mpg)
#cor(hp,mpg) 
#cor(cyl,mpg)
#cor(hp,mpg)^2 
#cor(cyl,mpg)^2 
#plot(hp,mpg,pch=cyl)
#lm.res = simple.lm(hp,mpg)


#page 127 ~ 128
data(chickwts) 
attach(chickwts) 
names(chickwts) 
boxplot(weight ~ feed)

our.mu = mean(weight) 
just.casein = weight[feed == "casein"] 
t.test(just.casein,mu = our.mu)

t.test(weight[feed == "casein"],weight[feed == "sunflower"])
#t.test(weight[feed == "linseed"],weight[feed == "soybean"], +var.equal=TRUE)
detach()


#page 129
make.t = function(x,mu) (mean(x)-mu)/( sqrt(var(x)/length(x)))
mu = 1;x=rnorm(100,mu,1) 
make.t(x,mu) 
mu = 10;x=rexp(100,1/mu);make.t(x,mu) 
results = c() 
for (i in 1:200) results[i] = make.t(rexp(100,1/mu),mu)

hist(results) # histogram looks bell shaped 
boxplot(results) # symmetric, not long-tailed 
qqnorm(results) 

for (i in 1:200) results[i] = make.t(rexp(8,1/mu),mu)
hist(results)
boxplot(results)
qqnorm(results) 

#page 130
