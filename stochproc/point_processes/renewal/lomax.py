import numpy as np
import hypothtst.tst.correlation.pt_processes as ppr
import matplotlib.pyplot as plt


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
    plt.plot(np.array([0,100,200,300,400]),\
            np.array([e_n1,e_n2,e_n3,e_n4,e_n5]))
    plt.show()


def p_vals_lomax_renewal(theta=10.0,k=2,n_sim=3000,null=False,\
                    window=4,intr_strt=400,intr_end=500, split_p=0.5,
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
            toss = np.random.uniform()<split_p
            if toss and t>intr_strt and t<intr_end:
                ts1.append(t)
            elif t>intr_strt and t<intr_end:
                ts2.append(t)
        e_n+=len(ts2)/n_sim
        ts1=np.array(ts1); ts1-=intr_strt
        ts2=np.array(ts2); ts2-=intr_strt
        p_val = ppr.correlation_score(ts1,ts2,window,(intr_end-intr_strt),\
                        verbose=verbose)
        p_vals.append(p_val)
    print("mean for second process:"+str(e_n))
    return np.array(p_vals)

