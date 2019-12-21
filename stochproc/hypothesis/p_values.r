dat = read.csv('C:\\Users\\ropandey\\Documents\\vsts\\Compute-Insights-optimum_settings\\scores.csv',sep=',',header=TRUE)

dat$p_value=rep(1,nrow(dat))

for (i in 1:length(dat$BadWithFeatureCount)){
    x=c(dat$BadWithFeatureCount[i],dat$GoodWithFeatureCount[i])
    n=c(dat$BadCount[i],dat$GoodCount[i])
    dat$p_value[i] = prop.test(x,n,alternative='greater')$p.value
}

write.csv(dat,file='C:\\Users\\ropandey\\Documents\\vsts\\Compute-Insights-optimum_settings\\scores2.csv')

