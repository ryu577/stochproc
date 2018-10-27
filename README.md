# Introduction

This repository aims to model and uncover the properties of all kinds of stochastic processes (processes
that are based on some kind of underlying phenomenon like a coin toss for which we can't know the outcome
for sure).

Some of these are taken from the book, Introduction to probability models by Sheldon Ross.

The best way to demonstrate the capabilities of this library is to describe some stochastic processes
it can model and show how to use it to extract some of their properties.


# Coin toss sequences

Let's say you and I start tossing fair coins. What is the probability you'll reach three consecutive heads before
I reach two consecutive heads?



```python
# To install the library, pip install stochproc from command line.
# hosted at - https://github.com/ryu577/stochproc
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


