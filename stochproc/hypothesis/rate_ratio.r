library("rateratio.test")

n=10
m=10

rateratio.test(c(9,3),c(n,m),alternative="greater")

binom.test(9,3+9,p=n/(n+m),alternative="greater")

