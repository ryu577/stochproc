
for (i in 1:9){
    q = quantile(c(1,2,3,4,5),.6,type=i)
    print(paste("Method-", toString(i), ": ", toString(q),sep=""))
}

