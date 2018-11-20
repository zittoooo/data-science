#변수 생성
a <- 1
x <- 3
a

print(a)
print(x)

# c() 함수로 벡터 생성
x <- c("fee","fie","foe", "fum")
print(x)

c("Everyone","loves", "stats.")

c(1,1,2,3,5,8,13,21)

#함수 내에서 연산을 할 수 있다.
c(1*pi, 2* pi, 3*pi, 4*pi)

#맞고 틀림의 논리형 값이다. ""가 필요없다.
c(TRUE, TRUE, FALSE, TRUE)


# v1와v2 변수를 c()를 이용해 결합할 수 있다.
v1 <- c(1,2,3)
v2 <- c(4,5,6)
v3 <- c("A","B","C")

# 수열
# n:m 표현식을 사용해 간단한 수열을 생성한다.
1:5
b <- 2:10
b
10:19
#큰 숫자부터 작은 숫자 순으로 수열을 만들 때에는 큰 숫자부터 써준다.
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

#rep  똑같은 값이 단순 반복되는 특별한 수열은
# 인자를 반복하는 rep 함수를 사용한다.  
rep (1, times =5)

# times 옵션을 넣으면 벡터의 값을 n번씩 반복함.
rep(1:2, each =2)

c <- 1:5
c

rep(c,5)
rep(c, each=5)


#numeric은 숫자 형태의 데이터다. 정수인 integer와 소수점이 있는 double이 있다.
a <- 3
a

#character는 문자 형태의 데이터다.
b <- "Charcter"
b

A <- c("a", "b", "c")
A
paste("a","b", sep="")

paste(A, c("d", "e"))

#문자와 숫자를 결합해 문자열 벡터에 저장할 수 있다.
f <- paste(A, 10)
f

#위의 모든 기능을 결합한 예제이다.
paste(A, 10, sep ="")
#자동으로 변수명을 만들 때 유용한 기능이다.
paste(A, 1:10, sep ="")

paste("Everybody", "loves", "cats.")
#sep 옵션을 이용하여 문자열 사이에 '-'를 넣었다.
paste("Everybody", "loves", "cats.", sep="-")
#여백을 제거하였다.
paste("Everybody", "loves", "cats.", sep="")


#substr(문자열, 시작, 끝)
ss <- c(" Moe", "Larry", "Curly")
substr("BigDataAnalysis", 1, 4)

#각 문자열의 첫 세글자 추출
substr(ss, 1,3)

#TRUE와 FALSE는 문자로 되어 있지만 큰 따옴표를 써주지 않아도
#오류가 나지않고 인식된다. 또한 F,T로 써도 된다.
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
#matrix( 이름, 행수, 열수)
#dim() : 행렬의 행과 열 수를 반환한다.
# 2x3 행렬을 만들어보자.
theData <- c(1.1, 1.2, 2.1, 2.2, 3.1, 3.2)
mat <- matrix(theData, 2, 3)
mat

dim(mat)

mat

#전치행렬
t(mat)

#mat과 mat 전치행렬의 곱
mat%*%t(mat)


#n차 대각선 행렬
diag(mat)

#행렬의 행과 열에 이름을 붙여보자.
mat
colnames(mat) <- c("IBM", "MSFT", "GOOG")
rownames(mat) <- c("IBM", "MSFT")
mat

mat
mat[1,]  #첫째 행 조회
mat[,3]  #셋째 열

A <- matrix(0,4,5)
A

A <- matrix(1:20, 4, 5)
A

A[c(1,4), c(2,3)]
A

A[c(1,4), c(2,3)] <- 1
A

A+1



#리스트 생성
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

#40보다 큰 리스트만 조회
score1[score1>40]
score1>40
score1[c(FALSE, FALSE, FALSE, FALSE, TRUE)]

#리스트 형식으로 합치기
score12 <- list(score1, score2)
score12

#합쳐진 리스트들을 여러 경우로 조회하기
score12[1]
score12[[2]]
score12[[1]][1]
score12[[1]][2]

#리스트 형식 해제  //
unlist(score1)
unlist(score2)
unlist(score12)


#여러 개의 열로 정리된 데이터를 데이터프레임으로 조립해보자
a <- c(1, 2, 4, 6, 3, 4)
b <- c(6, 4, 2, 4, 3.2, 4)
c <- c(7, 6, 4, 2, 5, 6)
d <- c(2, 4, 3, 1, 5, 6)
e <- data.frame(a, b, c, d)
e


#r에 내장된 iris 데이터프레임을 이용해 새로운 행과 열을 추가해보자.
data(iris)
head(iris)

#조건을 새로 만들어 newRow 변수에 내용을 할당한다.
newRow <- data.frame(Sepal.Length = 3.0, Sepal.Width = 3.2, Petal.Length = 1.6, Petal.Width = 0.3, Species = "newsetosa")
newRow

iris <- rbind(iris, newRow)
iris
dim(iris)

newcol <- 1:151
cbind(iris, newcol)




#새로운 데이터프레임을 만들어 보자
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

#merge 함수를 사용해 하나의 공통된 열로 데이터프레임을 병합해보자.
name <- c("Moe", "Larry", "Curly", "Harry")
year.born <- c(1887, 1982, 1983, 1964)
place.born <- c("BensonHurst", "Philadelphia", "Brooklyn", "Moscow")

#공통 열인 name을 갖는 born과 died 라는 두 개의 데이터프레임을 만들었다.
born <- data.frame(name, year.born, place.born)
born

name <- c("Curly", "Moe", "Larry")
year.died <- c(1952, 1975, 1975)
died <- data.frame(name, year.died)
died


merge(born, died, by = "name")


# R에 내장된 데이터세트인 mtcars를 조회해보자.

data(mtcars)
head(mtcars)

#각 데이터의 열 네임을 조회했다.
colnames(mtcars)

#1행부터 5행까지 mpg, cyl 변수의 열만 조회했다.
mtcars[1:5, c("mpg", "cyl") ]

mtcars[(mtcars$gear > 3 ) & (mtcars$cyl > 7| mtcars$mpg > 21), c("mpg", "cyl", "gear")]


#벡터에 있는 원소 선택
fib <- c(0, 1, 1, 2, 3, 5, 8, 13, 21, 34)
fib
fib[1]
fib[3]
#원소 1에서 3까지를 선택한다.
fib[1:3]

#1,2,4,8번째 원소를 선택한다.
fib[c(1, 2, 4, 8)]
#첫번 째 값을 제외하고 다른 모든 값을 반환한다.
fib[-1]

#첫번째에서 세 번째 값을 제외하고 모든 값을 반환한다.
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

# c>8 과 c<3 의 조건을 or연산한 값을 가져오는 것과 같은 표현이다.
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

#오늘의 날짜 조회
Sys.Date()
as.Date("2018-10-04")
as.Date("04/10/2018")
as.Date("10/04/2018", format = "%m/%d/%y")

#날짜를 문자열로 변환
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

#기본적인 통계량을 구해보자
x <- c(0, 1, 2, 3, 5, 8, 13, 21, 34)
y <- log(x+1)
y

mean(x)
median(x)
sd(x)
var(x)
cor(x,y)

#벡터의 기본연산
c <- 1:10
c
1/c
c^2
c^2 +1
log(c)

#벡터c에 log 라는 함수를 적용해 결과값 산출
sapply(c,log)

#벡터 2개를 이용한 연산
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

#log(c) < 2 인 조건을 만족하는 index에 대해 3을 대입하는 방식을 사용했다.
#이러한 표현 방식은 데이터 핸들링에서 많이 사용된다. 
c[log(c) < 2]  <- 3
c

#벡터 c의 length 속석에 20을 강제로 c 벡터에 20개의 변수를 저장한다.
#그러나 실제로 값이 있는 곳을 제외하면 NA가 들어간다.
length(c) <- 20
c
c[25] <- 1
c
length(c) <- 10
c


#유용한 기타 함수
#데이터를 저장하고 출력하는 방법

a <- c(1, 2, 3, 4, 5)
write.csv(a, "test.csv")  #변수를 csv파일로 저장.

b <- read.csv("test.csv")  #csv로 된 파일을 R로 읽어 온다.
save(a,file = "test.Rdata") #R데이터 파일로 저장한다.
a<-0

load("test.Rdata")  #R 데이터를 읽어 들인다.
a
print(a)



rm(a)  #데이터를 삭제한다.

ls()
rm(list = c("a"))
ls()
rm(list = ls(all =TRUE))
ls()

