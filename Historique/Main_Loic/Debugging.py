# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:13:36 2018

@author: Loic
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x)
                )  #On choisit cette sigmoide car la dérivée est simple


def couche(inputs, weight, bias):
    prod = np.dot(weight, inputs) + bias
    res = sigmoid(prod)
    return res


def reseau(inputs, total_weight, total_bias):
    n = len(total_weight)
    res = [inputs]
    for i in range(n):
        ouptut = couche(res[i], total_weight[i], total_bias[i])
        res.append(ouptut)
    return res


def erreur(th_output, outputs):  #Inutile dans le programme actuel
    ouput = outputs[-1]
    res = 0
    for i in range(len(ouput)):
        res = res + (th_output[i] - ouput[i][0])**2
    #print("Objectif: " + str(th_output[i]))
    #print("Resultat: " + str(ouput[i][0]))
    #print("Erreur " + str(res))
    return res


def Stochastic(total_inputs, total_ouputs, ini_weight, ini_bias, vitesse):
    n = len(total_inputs)
    W = ini_weight
    B = ini_bias
    E = []
    for i in range(n):
        I = total_inputs[i]
        O = total_ouputs[i]
        R = erreur(O, reseau(I, W, B))
        print(R)
        E.append(R)
        #(gW,gB) = gradient(I,W,B,O,R)
        #W = [W[i] - vitesse*gW[i] for i in range(len(W))]
        #B = [B[i] - vitesse*gB[i] for i in range(len(B))]
    return (W, B, E)


N = 20000

J = [
    np.array([[0.70710678], [0.70710678]]),
    np.array([[0.70710678], [-0.70710678]]),
    np.array([[-0.70710678], [-0.70710678]]),
    np.array([[-0.70710678], [0.70710678]]),
    np.array([[0.70710678], [0.70710678]])
]
O = [np.array([0]), np.array([1]), np.array([0]), np.array([1]), np.array([0])]
nW = [
    np.array([[-4.65191674, 5.09321415], [5.81228623, 5.01137582],
              [-4.70461299, 5.1405488], [-4.56017898, -5.36343531],
              [-4.55115374, -5.3538956]]),
    np.array(
        [[-0.94831607, -7.99318683, -0.05193885, -3.85915862, -4.67031686]])
]
nB = [
    np.array([[3.81001324], [-3.89113051], [3.84423457], [-3.91780908],
              [-3.91152002]]),
    np.array([[4.89001659]])
]
(nW, nB, E) = Stochastic(J, O, nW, nB, 0.05)
#plt.plot(E)
plt.plot(E[::4])
plt.plot(E[1::4])
plt.plot(E[2::4])
plt.plot(E[3::4])
