import numpy as np

def identify(x):
    return x

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x-c)
    sum_exp_a = np.sum(exp_a)
    
    y = exp_a / sum_exp_a
    
    return y  