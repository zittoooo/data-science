#���� ����
a <- 1
x <- 3
a

print(a)
print(x)

# c() �Լ��� ���� ����
x <- c("fee","fie","foe", "fum")
print(x)

c("Everyone","loves", "stats.")

c(1,1,2,3,5,8,13,21)

#�Լ� ������ ������ �� �� �ִ�.
c(1*pi, 2* pi, 3*pi, 4*pi)

#�°� Ʋ���� ������ ���̴�. ""�� �ʿ����.
c(TRUE, TRUE, FALSE, TRUE)


# v1��v2 ������ c()�� �̿��� ������ �� �ִ�.
v1 <- c(1,2,3)
v2 <- c(4,5,6)
v3 <- c("A","B","C")

# ����
# n:m ǥ������ ����� ������ ������ �����Ѵ�.
1:5
b <- 2:10
b
10:19
#ū ���ں��� ���� ���� ������ ������ ���� ������ ū ���ں��� ���ش�.
9:0
e <- 10:2
e


seq(from=0, to =20, by =2)

seq(from=0 , to =20, length.out =5)


seq(from=1.0, to = 2.0, length.out =5)
seq(0, 10, by=1)
seq(0, 10,length =20)

n <- 0
1:n

#rep  �Ȱ��� ���� �ܼ� �ݺ��Ǵ� Ư���� ������
# ���ڸ� �ݺ��ϴ� rep �Լ��� ����Ѵ�.  
rep (1, times =5)

# times �ɼ��� ������ ������ ���� n���� �ݺ���.
rep(1:2, each =2)

c <- 1:5
c

rep(c,5)
rep(c, each=5)


#numeric�� ���� ������ �����ʹ�. ������ integer�� �Ҽ����� �ִ� double�� �ִ�.
a <- 3
a

#character�� ���� ������ �����ʹ�.
b <- "Charcter"
b

A <- c("a", "b", "c")
A
paste("a","b", sep="")

paste(A, c("d", "e"))

#���ڿ� ���ڸ� ������ ���ڿ� ���Ϳ� ������ �� �ִ�.
f <- paste(A, 10)
f

#���� ��� ����� ������ �����̴�.
paste(A, 10, sep ="")
#�ڵ����� �������� ���� �� ������ ����̴�.
paste(A, 1:10, sep ="")

paste("Everybody", "loves", "cats.")
#sep �ɼ��� �̿��Ͽ� ���ڿ� ���̿� '-'�� �־���.
paste("Everybody", "loves", "cats.", sep="-")
#������ �����Ͽ���.
paste("Everybody", "loves", "cats.", sep="")


#substr(���ڿ�, ����, ��)
ss <- c(" Moe", "Larry", "Curly")
substr("BigDataAnalysis", 1, 4)

#�� ���ڿ��� ù ������ ����
substr(ss, 1,3)

#TRUE�� FALSE�� ���ڷ� �Ǿ� ������ ū ����ǥ�� ������ �ʾƵ�
#������ �����ʰ� �νĵȴ�. ���� F,T�� �ᵵ �ȴ�.
c <- TRUE
c
d <- T
d
e
f <- F
f

a <- 3
a == pi
a != pi

a < pi
a > pi
a <= pi
a >= pi

a = pi
a == pi


#Matrix
#matrix( �̸�, ���, ����)
#dim() : ����� ��� �� ���� ��ȯ�Ѵ�.
# 2x3 ����� ������.
theData <- c(1.1, 1.2, 2.1, 2.2, 3.1, 3.2)
mat <- matrix(theData, 2, 3)
mat

dim(mat)

mat

#��ġ���
t(mat)

#mat�� mat ��ġ����� ��
mat%*%t(mat)


#n�� �밢�� ���
diag(mat)

#����� ��� ���� �̸��� �ٿ�����.
mat
colnames(mat) <- c("IBM", "MSFT", "GOOG")
rownames(mat) <- c("IBM", "MSFT")
mat

mat
mat[1,]  #ù° �� ��ȸ
mat[,3]  #��° ��

A <- matrix(0,4,5)
A

A <- matrix(1:20, 4, 5)
A

A[c(1,4), c(2,3)]
A

A[c(1,4), c(2,3)] <- 1
A

A+1



#����Ʈ ����
list <- list(3.14, "MOe", c(1,1,2,3) , mean)
list

a <- 1:10
b<- matrix(1:10, 2, 5)
c <- c("name1", "name2")

alst <- list(x = a, y = b, z =c)
alst

alst$x
blst <- list(d=2: 10*10)
blst

alst$a

alst[[1]]

alst[1][2]



alst[[2]]

ablst <- c(alst, blst)
ablst


score1 <- list(10,20,30,40,50)
score2 <- list(c("a", "b"))

#40���� ū ����Ʈ�� ��ȸ
score1[score1>40]
score1>40
score1[c(FALSE, FALSE, FALSE, FALSE, TRUE)]

#����Ʈ �������� ��ġ��
score12 <- list(score1, score2)
score12

#������ ����Ʈ���� ���� ���� ��ȸ�ϱ�
score12[1]
score12[[2]]
score12[[1]][1]
score12[[1]][2]

#����Ʈ ���� ����  //
unlist(score1)
unlist(score2)
unlist(score12)


#���� ���� ���� ������ �����͸� ���������������� �����غ���
a <- c(1, 2, 4, 6, 3, 4)
b <- c(6, 4, 2, 4, 3.2, 4)
c <- c(7, 6, 4, 2, 5, 6)
d <- c(2, 4, 3, 1, 5, 6)
e <- data.frame(a, b, c, d)
e


#r�� ����� iris �������������� �̿��� ���ο� ��� ���� �߰��غ���.
data(iris)
head(iris)

#������ ���� ����� newRow ������ ������ �Ҵ��Ѵ�.
newRow <- data.frame(Sepal.Length = 3.0, Sepal.Width = 3.2, Petal.Length = 1.6, Petal.Width = 0.3, Species = "newsetosa")
newRow

iris <- rbind(iris, newRow)
iris
dim(iris)

newcol <- 1:151
cbind(iris, newcol)




#���ο� �������������� ����� ����
name <- c("john", "peter", "jennifer")
gender <- factor(c("m", "m", "f"))
hw1 <- c(60,60,80)
hw2 <- c(40,50,30)

grades <- data.frame(name, gender, hw1, hw2)
grades


grades[1,2]
grades[, "name"]
grades$name
grades[grades$gender == "m", ]
grades[, "hw1"]


data(iris)
head(iris)
subset(iris, select = Species, subset = (Petal.Length > 1.7))


subset(iris, select=c(Sepal.Length, Petal.Length, Species), subset = c(Sepal.Width == 3.0 & Petal.Width == 0.2))

head(with(iris, Species))

#merge �Լ��� ����� �ϳ��� ����� ���� �������������� �����غ���.
name <- c("Moe", "Larry", "Curly", "Harry")
year.born <- c(1887, 1982, 1983, 1964)
place.born <- c("BensonHurst", "Philadelphia", "Brooklyn", "Moscow")

#���� ���� name�� ���� born�� died ��� �� ���� �������������� �������.
born <- data.frame(name, year.born, place.born)
born

name <- c("Curly", "Moe", "Larry")
year.died <- c(1952, 1975, 1975)
died <- data.frame(name, year.died)
died


merge(born, died, by = "name")


# R�� ����� �����ͼ�Ʈ�� mtcars�� ��ȸ�غ���.

data(mtcars)
head(mtcars)

#�� �������� �� ������ ��ȸ�ߴ�.
colnames(mtcars)

#1����� 5����� mpg, cyl ������ ���� ��ȸ�ߴ�.
mtcars[1:5, c("mpg", "cyl") ]

mtcars[(mtcars$gear > 3 ) & (mtcars$cyl > 7| mtcars$mpg > 21), c("mpg", "cyl", "gear")]


#���Ϳ� �ִ� ���� ����
fib <- c(0, 1, 1, 2, 3, 5, 8, 13, 21, 34)
fib
fib[1]
fib[3]
#���� 1���� 3������ �����Ѵ�.
fib[1:3]

#1,2,4,8��° ���Ҹ� �����Ѵ�.
fib[c(1, 2, 4, 8)]
#ù�� ° ���� �����ϰ� �ٸ� ��� ���� ��ȯ�Ѵ�.
fib[-1]

#ù��°���� �� ��° ���� �����ϰ� ��� ���� ��ȯ�Ѵ�.
fib[-c(1:3)]


fib < 10
fib[fib <10]

fib %%2 ==0
fib[fib %% 2 == 0]

c<- 1:10
c
d <- 1:5
d[c(1, 3)]

c[c(2,3)]
d[c(1:3, 5)]
c[c>5 & c< 10]

# c>8 �� c<3 �� ������ or������ ���� �������� �Ͱ� ���� ǥ���̴�.
c[as.logical((c > 8) + (c < 3))]


years <- c(1960, 1964, 1976, 1994)
names(years) <- c("Kennedy", "Johnson", "Carter","Clinton")
years

years["Carter"]
years["Clinton"]

as.numeric("3.14")
as.integer(3.14)
as.numeric("foo")

as.character(101)
as.numeric(FALSE)
as.numeric(TRUE)

#������ ��¥ ��ȸ
Sys.Date()
as.Date("2018-10-04")
as.Date("04/10/2018")
as.Date("10/04/2018", format = "%m/%d/%y")

#��¥�� ���ڿ��� ��ȯ
as.Date("10/04/2018", format = "%m/%d/%y")
format(Sys.Date())
as.character(Sys.Date())
format(Sys.Date(), format = "%m/%d/%y")

format(Sys.Date(), '%a')
format(Sys.Date(), '%b')
format(Sys.Date(),'%B')
format(Sys.Date(), '%d')
format(Sys.Date(),'%y')
format(Sys.Date(), '%Y')

#�⺻���� ��跮�� ���غ���
x <- c(0, 1, 2, 3, 5, 8, 13, 21, 34)
y <- log(x+1)
y

mean(x)
median(x)
sd(x)
var(x)
cor(x,y)

#������ �⺻����
c <- 1:10
c
1/c
c^2
c^2 +1
log(c)

#����c�� log ��� �Լ��� ������ ����� ����
sapply(c,log)

#���� 2���� �̿��� ����
c <- 1:10
c
d <- (1:10) * 10
d
c + d

c * d

c ^ d

d ^ c

var(c)
log(c)
sum( (c - mean(c))^2) / (length(c) -1 )

c <- 1:10
c[log(c) < 2]

#log(c) < 2 �� ������ �����ϴ� index�� ���� 3�� �����ϴ� ����� ����ߴ�.
#�̷��� ǥ�� ����� ������ �ڵ鸵���� ���� ���ȴ�. 
c[log(c) < 2]  <- 3
c

#���� c�� length �Ӽ��� 20�� ������ c ���Ϳ� 20���� ������ �����Ѵ�.
#�׷��� ������ ���� �ִ� ���� �����ϸ� NA�� ����.
length(c) <- 20
c
c[25] <- 1
c
length(c) <- 10
c


#������ ��Ÿ �Լ�
#�����͸� �����ϰ� ����ϴ� ���

a <- c(1, 2, 3, 4, 5)
write.csv(a, "test.csv")  #������ csv���Ϸ� ����.

b <- read.csv("test.csv")  #csv�� �� ������ R�� �о� �´�.
save(a,file = "test.Rdata") #R������ ���Ϸ� �����Ѵ�.
a<-0

load("test.Rdata")  #R �����͸� �о� ���δ�.
a
print(a)



rm(a)  #�����͸� �����Ѵ�.

ls()
rm(list = c("a"))
ls()
rm(list = ls(all =TRUE))
ls()
