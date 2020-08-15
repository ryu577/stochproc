import numpy as np
from scipy.stats import lomax, binom_test
import matplotlib.pyplot as plt


def critical_interval(ts1,w,delt):
    t_end_prev=0
    critical=0
    for t1 in ts1:
        t_start=min(t1,t1+w)
        t_start=max(t_start,0)
        t_end=max(t1,t1+w)
        t_end=min(t_end,delt)
        t_start=max(t_start,t_end_prev)
        critical+=(t_end-t_start)
        t_end_prev=t_end
    return critical


def critical_events(ts1,ts2,w):
    """
    ts2 rains down upon ts1.
    """
    if len(ts1)==0 or len(ts2)==0:
        return 0
    j=0; critical=0
    for t in ts2:
        #First time an entry in ts1 crosses
        #current entry of ts2
        # ts1[j-1].....t......ts1[j]
        while j<len(ts1) and t>ts1[j]:
            j+=1
        if w>0 and j>0:
            critical+=ts1[j-1]+w>t
        elif j<len(ts1) and w<0:
            critical+=ts1[j]+w<t
    return critical


def correlation_score(ts1,ts2,w,delt,verbose=False):
    interv = critical_interval(ts1,w,delt)
    evnts = critical_events(ts1,ts2,w)
    if verbose:
        print(str(evnts)+","+str(len(ts2))+","+str(interv/delt))
    return binom_test(evnts,len(ts2),interv/delt,alternative='greater')


def p_vals_lomax_renewal(theta=10.0,k=2,n_sim=3000,null=False,\
                    window=4,intr_strt=400,intr_end=500,
                    verbose=False):
    p_vals = []
    e_n=0        
    for _ in range(n_sim):
        t=0
        ts1 = []; ts2 = []
        while t<intr_end:
            ## Move this lambda condition outside the while
            ## if you want the mixed Poisson. This currently
            ## is the Lomax renewal process.
            if null:
                lm = (k-1)/theta
            else:
                lm = np.random.gamma(k,1/theta)
            t += np.random.exponential(1/lm)
            toss = np.random.uniform()<0.5
            if toss and t>intr_strt and t<intr_end:
                ts1.append(t)
            elif t>intr_strt and t<intr_end:
                ts2.append(t)
        e_n+=len(ts2)/n_sim
        ts1=np.array(ts1); ts1-=intr_strt
        ts2=np.array(ts2); ts2-=intr_strt
        p_val = correlation_score(ts1,ts2,window,(intr_end-intr_strt),\
                        verbose=verbose)
        p_vals.append(p_val)
    print("mean:"+str(e_n))
    return np.array(p_vals)


def rejectn_rate(p_vals):
    alpha_hats = np.arange(0,1.00001,0.00001)
    alphas = np.zeros(len(alpha_hats))
    for p_val in p_vals:
        alphas+=(p_val<=alpha_hats)/len(p_vals)
    return alphas

def get_p_vals():
    p_vals_nul = p_vals_lomax_renewal(theta=3,k=3,null=True,window=.1)
    #(theta:1,corln:0.807);(2,0.6);(3,0.44);(4,0.37);(5,0.28) 
    p_vals_alt1 = p_vals_lomax_renewal(theta=1,k=1/1.5+1,window=.1)
    p_vals_alt2 = p_vals_lomax_renewal(theta=2,k=2/1.5+1,window=.1)
    p_vals_alt3 = p_vals_lomax_renewal(theta=3,k=3,window=.1)


def plot_pvals(p_vals_nul, p_vals_alt1,p_vals_alt2,p_vals_alt3):
    plt.hist(p_vals_nul,alpha=0.5,label="null", histtype='step',fill=False,stacked=True,color='green')
    plt.hist(p_vals_alt1,alpha=0.5,label="theta:0.3", histtype='step',fill=False,stacked=True,color='red')
    plt.hist(p_vals_alt2,alpha=0.5,label="theta:0.2", histtype='step',fill=False,stacked=True,color='orange')
    plt.hist(p_vals_alt3,alpha=0.5,label="theta:0.1", histtype='step',fill=False,stacked=True,color=(251/255, 206/255, 177/255))
    #plt.hist([p_vals_nul,p_vals_alt1,p_vals_alt2],alpha=0.5,label="theta:0.2",density=True, histtype='bar')
    plt.legend()
    plt.show()


def plot_power(p_vals_nul, p_vals_alt1,p_vals_alt2,p_vals_alt3):
    alphas = rejectn_rate(p_vals_nul)
    power1 = rejectn_rate(p_vals_alt1)
    power2 = rejectn_rate(p_vals_alt2)
    power3 = rejectn_rate(p_vals_alt3)
    plt.plot(alphas,power1,label="theta:0.3")
    plt.plot(alphas,power2,label="theta:0.2")
    plt.plot(alphas,power3,label="theta:0.1")
    plt.plot([0, 1], [0, 1], 'k-', lw=2)
    plt.legend()
    plt.show()


########################################
## Functional tests

def tst_critical_interv():
    ##### For critical interval.
    res = critical_interval([1,2,3],1,4)
    print(res==3)
    res = critical_interval([1,3],-5,4)
    print(res==3)
    res = critical_interval([1,3],-5,4)
    print(res==3)
    res = critical_interval([1,3],1,4)

def tst_critical_evnts():
    ##### For critical events.
    res = critical_events([1,2,3],[.5,1.5,2.5],.5)
    print(res==0)
    res = critical_events([1,2,3],[.5,1.5,2.5],1)
    print(res==2)
    res = critical_events([1,2,3],[.5,1.5,2.5],-1)
    print(res==3)


## Some tests on correlated point processes
def tst_sim_renewal_process():
    k=0.8
    lmb=2.0
    s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
    n_sim=5000
    for _ in range(n_sim):
        intervals = lomax.rvs(c=k, scale=(1/lmb), size=800)
        #intervals = np.random.exponential(scale=1,size=1000)
        time_stamps = np.cumsum(intervals)
        n1 = sum((time_stamps>400) * (time_stamps<900))
        n2 = sum((time_stamps>900) * (time_stamps<1400))
        s_n1n2+=n1*n2
        s_n1_sq+=n1*n1
        s_n2_sq+=n2*n2
        s_n1+=n1
        s_n2+=n2

    cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
    v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
    v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
    corln = cov/np.sqrt(v_n1*v_n2)
    print("correlation: " +str(corln))

#####

def tst_sim_2():
    k=0.8
    lmb=2.0

    s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
    n_sim=5000
    for _ in range(n_sim):
        intervals1 = lomax.rvs(c=k, scale=(1/lmb), size=800)
        intervals2 = lomax.rvs(c=k, scale=(1/lmb), size=800)
        #intervals = np.random.exponential(scale=1,size=1000)
        time_stamps1 = np.cumsum(intervals1)
        time_stamps2 = np.cumsum(intervals2)
        n1 = sum((time_stamps1>400) * (time_stamps1<900))
        n2 = sum((time_stamps2>400) * (time_stamps2<900))
        s_n1n2+=n1*n2
        s_n1_sq+=n1*n1
        s_n2_sq+=n2*n2
        s_n1+=n1
        s_n2+=n2
    cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
    v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
    v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
    corln = cov/np.sqrt(v_n1*v_n2)
    print("correlation: " +str(corln))


#####

def tst_sim_3(k=7.0,theta=0.5):
    s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
    n_sim=5000
    for _ in range(n_sim):
        intervals = lomax.rvs(c=k, scale=theta, size=2000)
        #intervals = np.random.exponential(scale=1,size=1000)
        time_stamps = np.cumsum(intervals)
        bi_furcator = np.random.choice(2,size=len(time_stamps))
        time_stamps1 = time_stamps[bi_furcator==1]
        time_stamps2 = time_stamps[bi_furcator==0]
        n1 = sum((time_stamps1>50) * (time_stamps1<90))
        n2 = sum((time_stamps2>50) * (time_stamps2<90))
        s_n1n2+=n1*n2
        s_n1_sq+=n1*n1
        s_n2_sq+=n2*n2
        s_n1+=n1
        s_n2+=n2
    cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
    v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
    v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
    corln = cov/np.sqrt(v_n1*v_n2)
    print("correlation: " +str(corln))

#####
## Check mean of Lomax renewal process

def lomax_renewal_correlation(k=2.0, theta=1.0):
    s_n1=0
    n_sim=5000
    for _ in range(n_sim):
        intervals = lomax.rvs(c=k, scale=theta, size=1200)
        #intervals = np.random.exponential(scale=1,size=1000)
        time_stamps = np.cumsum(intervals)
        #n1 = sum((time_stamps>100) * (time_stamps<200))
        n1 = sum(time_stamps<100)
        s_n1+=n1

    e_n1 = s_n1/n_sim

    print("simulated mean: " +str(e_n1))
    #print("actual mean-1: " +str(k*200/theta))
    print("actual mean-2: " +str((k-1)*200/theta))


#####
## Poisson mixture.. inducing correlation.

def mixed_poisson_correlation(k=2.0,theta=0.01):
    s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
    n_sim=3000
    for _ in range(n_sim):
        t=0; n1=0; n2=0
        lm = np.random.gamma(k,theta)
        #lm=1.2
        while t<130:
            t+=np.random.exponential(1/lm)
            toss = np.random.uniform()<0.5
            n1+=(t>100)*(t<110)*toss
            #n2+=(t>120)*(t<130)
            n2+=(t>100)*(t<110)*(1-toss)
        s_n1+=n1
        s_n2+=n2
        s_n1n2+=n1*n2
        s_n1_sq+=n1*n1
        s_n2_sq+=n2*n2

    e_n1 = s_n1/n_sim
    cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
    v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
    v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
    corln = cov/np.sqrt(v_n1*v_n2)
    print("Correlation: " + str(corln))
    print("Simulated mean: " + str(2*e_n1))
    print("Theoretical mean:" + str(k*10*theta))



def lomax_renewal_stats(theta=10,k=2,n_sim=3000,null=False,\
                    window=4,delt=500,verbose=False):
    e_n1=e_n2=e_n3=e_n4=e_n5=0
    s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
    for _ in range(n_sim):
        t=0; n1=n2=0
        while t<delt:
            ## Move this lambda condition outside the while
            ## if you want the mixed Poisson. This currently
            ## is the Lomax renewal process.
            if null:
                lm = k*theta
            else:
                lm = np.random.gamma(k,1/theta)
            t += np.random.exponential(1/lm)
            e_n1+=((t>0)*(t<100))/n_sim
            e_n2+=((t>100)*(t<200))/n_sim
            e_n3+=((t>200)*(t<300))/n_sim
            e_n4+=((t>300)*(t<400))/n_sim
            e_n5+=((t>400)*(t<500))/n_sim
            toss = np.random.uniform()<0.5
            n1+=(t>400)*(t<500)*toss
            n2+=(t>400)*(t<500)*(1-toss)
        s_n1+=n1
        s_n2+=n2
        s_n1n2+=n1*n2
        s_n1_sq+=n1*n1
        s_n2_sq+=n2*n2
    cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
    v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
    v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
    corln = cov/np.sqrt(v_n1*v_n2)
    print(corln)
    plt.plot(np.array([0,100,200,300,400]),np.array([e_n1,e_n2,e_n3,e_n4,e_n5]))
    plt.show()

def lomax_exponmix():
    #### Verify Lomax equivalence with exponential-mix.
    k=4; theta=0.1
    ## In numpy's definition, the scale, theta is inverse of Ross definition.
    lm = np.random.gamma(k,1/theta,size=1000)
    lomax_mix=np.random.exponential(1/lm)
    mean1=np.mean(lomax_mix)
    lomax_direct=lomax.rvs(c=k,scale=theta,size=1000)
    mean2=np.mean(lomax_direct)
    mean3 = theta/(k-1)


