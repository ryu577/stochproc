from scipy.stats import gamma
import numpy as np
from collections import Counter, OrderedDict
import time
from scipy.integrate import quad, dblquad

def tst_approaches():
    start = time.process_time()
    print("Run sampling based approach")
    pdfs = [gamma(a=5, scale=1.0/10), gamma(a=5, scale = 1.0/10), gamma(a=5, scale = 1.0/10)]
    sampleSize = 100000
    samples = np.array([pdf.rvs(size=sampleSize) for pdf in pdfs])
    max_idx = samples.argmax(axis=0)
    occurCount = Counter(max_idx)
    prob = [(key, occurCount[key]/sampleSize) for key in occurCount]
    end = time.process_time()
    sortedProb = OrderedDict(sorted(prob))
    print(sortedProb)
    print("execution time {} seconds".format(end - start))

    print("\nRun numerical integration based approach")
    start = time.process_time()
    if len(pdfs)==2:
        P_0_largest = quad(lambda x: pdfs[1].pdf(x) * pdfs[0].sf(x), 0, np.inf)[0]
        P_1_largest = 1-P_0_largest
        print({0:P_0_largest, 1:P_1_largest})

    if len(pdfs)==3:
        P_0_largest = dblquad(lambda x, y: pdfs[1].pdf(x) * pdfs[2].pdf(y) * pdfs[0].sf(max(x,y)), 0, np.inf, 0, np.inf)[0]
        P_1_largest = dblquad(lambda x, y: pdfs[0].pdf(x) * pdfs[2].pdf(x) * pdfs[1].sf(max(x,y)), 0, np.inf, 0, np.inf)[0]
        P_2_largest = dblquad(lambda x, y: pdfs[0].pdf(x) * pdfs[1].pdf(x) * pdfs[2].sf(max(x,y)), 0, np.inf, 0, np.inf)[0]
        # P_2_largest = 1-P_0_largest-P_1_largest
        print({0:P_0_largest, 1:P_1_largest, 2:P_2_largest})

    end = time.process_time()
    print("execution time {} seconds".format(end - start))

    start = time.process_time()
    dblquad(lambda x, y: pdfs[1].pdf(x) * pdfs[2].pdf(y) * pdfs[0].sf(max(x,y)), 0, np.inf, 0, np.inf)[0]
    end = time.process_time()
    print("execution time {} seconds".format(end - start))

    start = time.process_time()
    def fn1(x1):
        return quad(lambda x: pdfs[1].pdf(x) * pdfs[2].sf(x), x1, np.inf)[0]+\
            quad(lambda x: pdfs[2].pdf(x) * pdfs[1].sf(x), x1, np.inf)[0]

    quad(lambda x:fn1(x)*pdfs[0].pdf(x),0,np.inf)[0]
    end = time.process_time()
    print("execution time {} seconds".format(end - start))


