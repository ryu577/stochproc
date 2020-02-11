# Introduction

This repository aims to model and uncover the properties of all kinds of stochastic processes (processes
that are based on some kind of underlying phenomenon like a coin toss for which we can't know the outcome
for sure).

Some of these are taken from the book, Introduction to probability models by Sheldon Ross.

The best way to demonstrate the capabilities of this library is to describe some stochastic processes
it can model and show how to use it to extract some of their properties.

# Installation
To install the library, run (pyray was taken on pypi):

```
pip install stochproc
```

Make sure you have all the requirements (requirements.txt) installed. If not, you can run:

```
pip install -r requirements.txt
```

Alternately, you can fork/download the code and run from the main folder:

```
python setup.py install
```


# Machine reliability

Let's say we start up a machine. It stays up some random amount of time before succumbing to failure.
The amount of time it stays up is a random variable. This variable models the mean time between failures.

And once the machine is down, it takes a certain amount of time to get repaired. This is the mean time
to repair.


The question is at any time t, what is the probability the machine is up and running?

Chapter 11 of [1] provides a closed form. However, we want to validate this with simulation.

```python
from stochproc.reliability.machinerepair import *

for i in range(10):
	probs = []
	stds = []
	for t in range(1,100):
		prob, std = updown(t)
		probs.append(prob)
		stds.append(std)
	plt.plot(np.arange(1,100), probs,alpha=0.4,color='pink')

xs = np.arange(1,100)
plt.plot(xs, closed_form(xs),color='red')
plt.xlabel('Time')
plt.ylabel('Reliability of system')
plt.show()
```

This leads to the following plot.

<a href="https://medium.com/@rohitpandey576/coin-toss-markov-chains-7995cb303406" 
target="_blank"><img src="https://github.com/ryu577/stochproc/blob/master/plots/mcreliability.png" 
alt="Image formed by above method" width="240" height="240" border="10" /></a>


# Coin toss sequences

Let's say you and I start tossing fair coins. What is the probability you'll reach three consecutive heads before
I reach two consecutive heads? What about in general you reaching (n+1) consecutive heads before I reach n consecutive
heads?


```python
from stochproc.competitivecointoss.smallmarkov import *

ns = np.arange(2,15)
win_probs = []
for n in ns:
    # The losing markov sequence of coin tosses that needs (n-1) heads.
    lose_seq = MarkovSequence(get_consecutive_heads_mat(n))
    # The winning markov sequence of coin tosses that needs n heads.
    win_seq = MarkovSequence(get_consecutive_heads_mat(n+1))
    # If you multiply the two sequence objects, you get the probability
    # that the first one beats the second one.
    win_prob = win_seq*lose_seq
    win_probs.append(win_prob)

plt.plot(ns, win_probs)
```

This leads to the following plot:


<a href="https://medium.com/@rohitpandey576/coin-toss-markov-chains-7995cb303406" 
target="_blank"><img src="https://github.com/ryu577/ryu577.github.io/blob/master/Downloads/CompetitiveCoinToss/probs_with_n.png" 
alt="Image formed by above method" width="240" height="240" border="10" /></a>





