#!/usr/bin/env python3
#filename: NeuralNetDeepLearn_Ch1_BinaryConversion.py

import math
import numpy as np

# Old Output from neural net
oldoutput = np.array([
    [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99],
    ])

# Desired New Output
o = np.array([
    [0,0,0,0],
    [1,0,0,0],
    [0,1,0,0],
    [1,1,0,0],
    [0,0,1,0],
    [1,0,1,0],
    [0,1,1,0],
    [1,1,1,0],
    [0,0,0,1],
    [1,0,0,1],
    ])
#w = 110 * o.T
#w = 11 * o.T
w = 20 * o.T

ones = np.array([
    [1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1],
    ])
#b = -10 * ones
#b = -5.5 * ones
b = -10 * ones

z = w @ oldoutput + b

# use numpy built-in ufuncs
#def vectorized_sigmoid(x):
#    return 1/(1+np.exp(-x))
#newoutput = vectorized_sigmoid(z)
#print("{}\n".format(newoutput))
#print("{}\n".format(o.T))

def rnd_sigmoid(x):
    return round(1/(1+math.exp(-x)),2)
    #return round(1/(1+math.exp(-x)))
vectorized_rndsig = np.vectorize(rnd_sigmoid)

newoutput = vectorized_rndsig(z)


print("\n")
print("{}\n".format('The old output:'))
print("{}\n".format(oldoutput))
print("{}\n".format('z, before the sigmoid:'))
print("{}\n".format(z))
print("{}\n".format('The new output:'))
print("{}\n".format(newoutput))
print("{}\n".format('The desired new output:'))
print("{}\n".format(o.T))
print("{}\n".format('The difference between them:'))
print("{}\n".format(newoutput-o.T))
print("\n")
print("{}\n".format('If the difference is 0, then here is a solution set of weights and biases:'))
print("{}\n".format('Weights:'))
print("{}\n".format(w))
print("{}\n".format('Biases:'))
print("{}\n".format(b))
