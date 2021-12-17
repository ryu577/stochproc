import numpy as np
import matplotlib.pyplot as plt


# Let's simulate some non-renewal processes now.
def sim_alternating():
    """
    This is not a renewal process.
    """
    catches = 0
    for _ in range(100000):
        j = np.random.uniform()*1000
        # j = np.random.exponential(500)
        t_i = 0
        i = 0
        while t_i < j+100:
            if i % 2 == 0:
                t_i += 10
            else:
                t_i += 20
            if j < t_i and t_i < j+1:
                catches += 1
            i += 1
    print(catches/100000)


def sim_alternating_exp():
    """
    This is not a renewal process.
    """
    catches = 0
    for _ in range(100000):
        j = np.random.uniform()*1000
        # j = np.random.exponential(500)
        t_i = 0
        i = 0
        while t_i < j+100:
            if i % 2 == 1:
                t_i += np.random.exponential(10)
            else:
                t_i += np.random.exponential(20)
            if j < t_i and t_i < j+1:
                catches += 1
            i += 1
    print(catches/100000)


def sim_normal():
    catches = 0
    for _ in range(10000):
        j = np.random.uniform()*1000
        t_i = 0
        while t_i < j+500:
            t_i += np.random.normal(10,1)
            if j < t_i and t_i < j+1:
                catches += 1
    print(catches/10000)


def generate_cov(dim, _corr = 0.5):
    acc  = []
    for i in range(dim):
        row = np.ones((1,dim)) * _corr
        row[0][i] = 1
        acc.append(row)
    return np.concatenate(acc,axis=0)


def gen_corr_normal(n_dist=3, nn=5, corr=0.9, mu=10):
    acc = []
    for i in range(n_dist):
        acc.append(np.reshape(np.random.normal(0, 1, nn), (nn, -1)))

    # Compute
    all_norm = np.concatenate(acc, axis=1)
    cov = generate_cov(n_dist, corr)
    c = np.linalg.cholesky(cov)
    y = np.transpose(0 + np.dot(c, np.transpose(all_norm)))
    y += np.ones(y.shape)*mu
    return y


y = gen_corr_normal(corr=-0.45, nn=100)
plt.plot(y[::, 0], y[::, 1])
plt.title("Mean: " + str(np.mean(y)))
plt.show()

catches = 0
for i in range(10000):
    j = np.random.uniform()*1000
    t_i = 0
    if i % 60 == 0:
        rnd = gen_corr_normal(corr=-0.2, mu=10, nn=20, n_dist=3)
        rnd = rnd.flatten()
    while t_i < j+500:
        t_i += rnd[i % 60]
        if j < t_i and t_i < j+1:
            catches += 1
print(catches/10000)

