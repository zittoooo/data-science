#(1) Multiplication
A <- matrix(data = 1:36, nrow = 6) 
A
B <- matrix(data = 1:30, nrow = 6) 
B
A %*% B 

#(2) Hadamard Multiplication 
A <- matrix(data = 1:36, nrow = 6) 
A
B <- matrix(data = 11:46, nrow = 6)
B
A * B 

#(3) Dot Product 
X <- matrix(data = 1:10, nrow = 10) 
X
Y <- matrix(data = 11:20, nrow = 10) 
Y

#(4) Properties of Matrix Multiplication 
A <- matrix(data = 1:25, nrow = 5) 
B <- matrix(data = 26:50, nrow = 5) 
C <- matrix(data = 51:75, nrow = 5)
A %*% (B + C) 
A %*% B + A %*% C 

A <- matrix(data = 1:25, nrow = 5) 
B <- matrix(data = 26:50, nrow = 5) 
C <- matrix(data = 51:75, nrow = 5) 
(A %*% B) %*% C 
A %*% (B %*% C) 


A <- matrix(data = 1:25, nrow = 5) 
B <- matrix(data = 26:50, nrow = 5) 
A %*% B 
B %*% A 


#2) Matrix Transpose 
A <- matrix(data = 1:25, nrow = 5, ncol = 5, byrow = TRUE) 
A
t(A) 

#(1) Transpose Property 
A <- matrix(data = 1:25, nrow = 5) 
B <- matrix(data = 25:49, nrow = 5) 
t(A %*% B) 
t(B) %*% t(A) 


#3) Solve System of Linear Equations 
A <- matrix(data = c(1,3,2,4,2,4,3,5,1,6,7,2,1,5,6,7), nrow = 4, byrow = TRUE) 
A
B <- matrix(data = c(1, 2, 3, 4), nrow = 4) 
B
solve(a = A, b = B) 

#4) Identity Matrix
I <- diag(x = 1, nrow = 5, ncol = 5) 
I
A <- matrix(data = 1:25, nrow = 5) 
A %*% I 
I %*% A


#5) Matrix Inverse 
A <- matrix(data = c(1,2,3,1,2,3,4,5,6,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,3), nrow = 5)
A
library(MASS) 
ginv(A)
A %*% ginv(A) 
ginv(A) %*% A 


#6)  Solve System of Linear Equations Revisited 
A <- matrix(data = c(1, 3, 2, 4, 2, 4, 3, 5, 1, 6, 7, 2, 1, 5, 6, 7), nrow = 4, byrow = TRUE) 
A
B <- matrix(data = c(1, 2, 3, 4), nrow = 4) 
B
library(MASS) 
X <- ginv(A) %*% B 
X

#7) Determinant 
A <- matrix(data = c(1,3,2,4,2,4,3,5,1,6,7,2,1,5,6,7), nrow = 4, byrow = TRUE)
A
det(A)



#(1) Properties 
lpNorm(A = matrix(data = rep(0, 10)), p = 1) == 0 
lpNorm(A = matrix(data = 1:10) + matrix(data = 11:20), p = 1) <=    lpNorm(A = matrix(data = 1:10), p = 1) + lpNorm(A = matrix(data = 11:20), p = 1)

tempFunc <- function(i) {     lpNorm(A = i * matrix(data = 1:10), p = 1) == abs(i) * lpNorm(A = matrix(data = 1:10), p = 1)    } 
all(sapply(X = -10:10, FUN = tempFunc)) 


#(1) Diagonal Matrix 
A <- diag(x = c(1:5, 6, 1, 2, 3, 4), nrow = 10) 
A
X <- matrix
(data = 21:30)
X
A %*% X
library(MASS)
ginv(A)



A <- matrix(data = c(1, 2, 2, 1), nrow = 2) 
A
all(A == t(A)) 

lpNorm(A = matrix(data = c(1, 0, 0, 0)), p = 2) 


X <- matrix(data = c(11, 0, 0, 0)) 
Y <- matrix(data = c(0, 11, 0, 0)) 
all(t(X) %*% Y == 0) 


X <- matrix(data = c(1, 0, 0, 0)) 
Y <- matrix(data = c(0, 1, 0, 0)) 
lpNorm(A = X, p = 2) == 1 
lpNorm(A = Y, p = 2) == 1 
all(t(X) %*% Y == 0) 


A <- matrix(data = c(1, 0, 0, 0, 1, 0, 0, 0, 1), nrow = 3, byrow = TRUE)
A
all(t(A) %*% A == A %*% t(A)) 
all(t(A) %*% A == diag(x = 1, nrow = 3)) 
library(MASS) 
all(t(A) == ginv(A)) 

A <- matrix(data = 1:25, nrow = 5, byrow = TRUE) 
A
y <- eigen(x = A) 
library(MASS) 
all.equal(y$vectors %*% diag(y$values) %*% ginv(y$vectors), A) 



A <- matrix(data = 1:36, nrow = 6, byrow = TRUE) 
A
y <- svd(x = A) 
y 
all.equal(y$u %*% diag(y$d) %*% t(y$v), A) 


A <- matrix(data = 1:25, nrow = 5) 
A
B <- ginv(A) 
B
y <- svd(A) 
all.equal(y$v %*% ginv(diag(y$d)) %*% t(y$u), B)


#14) Trace    #REPLAY
A <- diag(x = 1:10)
A
library(psych)
tr(A) 



#2) More Case Studies 

library(MASS)
a <- matrix(c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,     0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1), 9, 4) 
print(a) 

a.svd <- svd(a) 
a.svd$d 
ds <- diag(1/a.svd$d[1:3])
u <- a.svd$u 
v <- a.svd$v 
us <- as.matrix(u[, 1:3]) 
vs <- as.matrix(v[, 1:3]) 
(a.ginv <- vs %*% ds %*% t(us)) 

ginv(a)


library(ReadImages)



library(foreign)
auto <- read.dta("http://statistics.ats.ucla.edu/stat/data/auto.dta") 
pca.m1 <- prcomp(~trunk + weight + length + headroom, data = auto,     scale = TRUE) 
screeplot(pca.m1) 
xvars <- with(auto, cbind(trunk, weight, length, headroom))
corr <- cor(xvars) 
a <- eigen(corr) 
(std <- sqrt(a$values)) 
(rotation <- a$vectors) 
# svd approach 
df <- nrow(xvars) - 1 
zvars <- scale(xvars) 
z.svd <- svd(zvars)
z.svd$d/sqrt(df)
z.svd$v


cnut <- read.dta("http://statistics.ats.ucla.edu/stat/data/cerealnut.dta") 
# centering the variables 
mds.data <- as.matrix(sweep(cnut[, -1], 2, colMeans(cnut[, -1]))) 
dismat <- dist(mds.data) 
mds.m1 <- cmdscale(dismat, k = 8, eig = TRUE) 
mds.m1$eig 
mds.m1 <- cmdscale(dismat, k = 2, eig = TRUE)
x <- mds.m1$points[, 1] 
y <- mds.m1$points[, 2] 
plot(x, y) 
text(x + 20, y, label = cnut$brand) 



# eigenvalues 
xx <- svd(mds.data %*% t(mds.data)) 
xx$d
# coordinates 
xxd <- xx$v %*% sqrt(diag(xx$d))
x1 <- xxd[, 1]
y1 <- xxd[, 2]

plot(x1, y1)
text(x1 + 20, y1, label = cnut$brand)



set.seed(123)
ss <- sample(1:50, 15)
df <- USArrests[ss, ]
df.scaled <- scale(df)
dist.eucl <- dist(df.scaled, method = "euclidean") 
round(as.matrix(dist.eucl)[1:3, 1:3], 1) 

library("factoextra")
dist.cor <- get_dist(df.scaled, method = "pearson") 
round(as.matrix(dist.cor)[1:3, 1:3], 1) 

library(cluster) 
data(flower)
head(flower, 3)

str(flower)
dd <- daisy(flower)
round(as.matrix(dd)[1:3, 1:3], 2)
library(factoextra)
fviz_dist(dist.eucl)


url <- "http://rosetta.reltech.org/TC/v15/Mapping/data/dist-Aus.csv" 
dist.au <- read.csv(url)
dist.au <- read.csv("dist-Aus.csv") 
dist.au 
row.names(dist.au) <- dist.au[, 1] 
dist.au <- dist.au[, -1] 
dist.au 

fit <- cmdscale(dist.au, eig = TRUE, k = 2)
x <- fit$points[, 1] 
y <- fit$points[, 2] 
plot(x, y, pch = 19, xlim = range(x) + c(0, 600)) 
city.names <- c("Adelaide", "Alice Springs", "Brisbane", "Darwin", "Hobart", "Melbourne", "Perth", "Sydney") 
text(x, y, pos = 4, labels = city.names)


x <- 0 - x
y <- 0 - y
plot(x, y, pch = 19, xlim = range(x) + c(0, 600))
text(x, y, pos = 4, labels = city.names)


install.packages("igraph")
library(igraph) 
g <- graph.full(nrow(dist.au)) 
V(g)$label <- city.names 
layout <- layout.mds(g, dist = as.matrix(dist.au)) 
plot(g, layout = layout, vertex.size = 3) 


data("swiss") 
head(swiss) 
library(magrittr) 
library(dplyr) 
library(ggpubr) 
mds <- swiss %>% dist() %>% cmdscale() %>%  as_tibble() 
colnames(mds) <- c("Dim.1", "Dim.2")

clust <- kmeans(mds, 3)$cluster %>% as.factor() 

mds <- mds %>%  mutate(groups = clust) 


library(magrittr)
library(dplyr)
library(ggpubr)
library(MASS)
mds <- swiss %>% dist() %>% isoMDS() %>% .$points %>% as_tibble() 
colnames(mds) <- c("Dim.1", "Dim.2") 



library(MASS) 
mds <- swiss %>% dist() %>% sammon() %>% .$points %>% as_tibble() 
colnames(mds) <- c("Dim.1", "Dim.2")



res.cor <- cor(mtcars, method = "spearman") 
mds.cor <- (1 - res.cor) %>% cmdscale() %>% as_tibble() 
colnames(mds.cor) <- c("Dim.1", "Dim.2") 


library(datasets)
data(USArrests)
summary(USArrests) 
myData <- USArrests
fit <- princomp(myData, cor=TRUE)
summary(fit)
fit$scores
biplot(fit) 
