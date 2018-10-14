# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:10:09 2018

@author: Loic
"""

import time
import random as rd

import numpy as np
import matplotlib.pyplot as plt
import idx2numpy as idx

from conf import train_images_path, train_labels_path
from network_base import front_prop, backprop, random_w_b, save_network, load_network

#%% Batch Training

def batch_training(L_inputs, L_th_output, reseau, weights, bias, rate, iterations):
    error = []
    for k in range(iterations):
        delta_weight = [weights[k]*0 for k in range(len(weights))]
        delta_bias = [bias[k]*0 for k in range(len(bias))]
        cost_tot = 0
        for data in range(len(L_inputs)):
            gw, gb, cost = backprop(L_inputs[data], L_th_output[data], reseau, weights, bias)
            for col in range(len(gw)):
                delta_weight[col] += gw[col]*rate/len(L_inputs)
                delta_bias[col] += gb[col]*rate/len(L_inputs)
            cost_tot += cost/len(L_inputs)
        error.append(cost_tot)
        for col in range(len(weights)):
            weights[col] += -delta_weight[col]
            bias[col] += -delta_bias[col]
    return weights, bias, error


# def minibatch_training(L_inputs, L_th_output, reseau, minibatch, weights=random_w_b(L_inputs[0],reseau)[0], bias=random_w_b(L_inputs[0],reseau)[1]):
#     pass 


#%% Stochastic learning

def stochastic_training(total_inputs, total_ouputs, ini_weight, ini_bias, vitesse, reseau, iterations=1):
    n = len(total_inputs)
    W = ini_weight
    B = ini_bias
    E = []
    start_time = time.time()
    for j in range(iterations):
        for i in range(n):
            I = total_inputs[i]
            O = total_ouputs[i]
            (gW, gB, ee) = backprop(I, O, reseau, W, B)
            E.append(ee)
            W = [W[i] - vitesse*gW[i] for i in range(len(W))]
            B = [B[i] - vitesse*gB[i] for i in range(len(B))]
            if int((i+n*j)/n/iterations*100) == (i+n*j)/n/iterations*100:
                ETA = round(((time.time()-start_time)*(n-i+(n*(iterations-j)))/(i+n*j))) if i > 0 else 0
                ETA = f"{ETA//3600}h {(ETA%3600)//60}min {ETA%60}sec" if ETA > 60 else f"{ETA//60}min {ETA%60}sec" if ETA > 60 else f"{ETA}sec"
                if i > 0:
                    print(f"{round((i+n*j)/n/iterations*100)}% done (ETA: {ETA})")
                # print(str(i/n*100*(j+1)/iterations) + " % done")
    return (W, B, E)


def traite_entrees(total_inputs): #It works maggle
    n = len(total_inputs)
    m = len(total_inputs[0])
    moy = np.array([0 for i in range(m)])
    for i in range(n):
        moy = moy + total_inputs[i]
    res = [total_inputs[i] - moy/n for i in range(n)]
    for i in range(n):
        norm = 0
        for j in range(m):
            norm += (res[i][j])**2
        norm = np.sqrt(norm)
        for j in range(m):
            res[i][j] = res[i][j]/norm
    return res


#%% Xor

def le_xor_batch(pas, reseau=[4, 1]):
    (Wt, Bt) = random_w_b([0, 0], reseau)
    N = 10000
    I = [np.array([int(i%4 == 0 or i%4 == 1), int(i%4 == 0 or i%4 == 3)]) for i in range(4)]
    O = [np.array([int((I[i][0] > 0 and I[i][1] <= 0) or (I[i][0] <= 0 and I[i][1] > 0))]) for i in range(4)]
    (nW, nB, E) = batch_training(I, O, reseau, Wt, Bt, pas, N)
    R11 = front_prop(I[0], reseau, nW, nB)
    R10 = front_prop(I[1], reseau, nW, nB)
    R00 = front_prop(I[2], reseau, nW, nB)
    R01 = front_prop(I[3], reseau, nW, nB)
    plt.plot(E)
    print("11: " + str(R11[-1]) + " ie. " + str(int(R11[-1][0] > .5)))
    print("10: " + str(R10[-1]) + " ie. " + str(int(R10[-1][0] > .5)))
    print("01: " + str(R01[-1]) + " ie. " + str(int(R01[-1][0] > .5)))
    print("00: " + str(R00[-1]) + " ie. " + str(int(R00[-1][0] > .5)))


def le_xor_stochastic(pas, N=20000, reseau=[4, 1]):
    (Wt, Bt) = random_w_b([0, 0], reseau)
    I = [np.array([int(i%4 == 0 or i%4 == 1), int(i%4 == 0 or i%4 == 3)]) for i in range(N)]
    I = traite_entrees(I)
    O = [np.array([int((I[i][0] > 0 and I[i][1] < 0) or (I[i][0] < 0 and I[i][1] > 0))]) for i in range(N)]
    (nW, nB, E) = stochastic_training(I, O, Wt, Bt, pas, reseau)
    R11 = front_prop(I[0], reseau, nW, nB)
    R10 = front_prop(I[1], reseau, nW, nB)
    R00 = front_prop(I[2], reseau, nW, nB)
    R01 = front_prop(I[3], reseau, nW, nB)
    save_network(reseau, nW, nB, "XOR.data")
    print(load_network('XOR.data'))
    plt.plot(E[::4])
    plt.plot(E[1::4])
    plt.plot(E[2::4])
    plt.plot(E[3::4])
    print("11: " + str(R11[-1]) + " ie. " + str(int(R11[-1][0] > .5)))
    print("10: " + str(R10[-1]) + " ie. " + str(int(R10[-1][0] > .5)))
    print("01: " + str(R01[-1]) + " ie. " + str(int(R01[-1][0] > .5)))
    print("00: " + str(R00[-1]) + " ie. " + str(int(R00[-1][0] > .5)))

#%% MNIST

def MNIST_datas():
    with open(train_images_path, "rb") as train_images:
        with open(train_labels_path, "rb") as train_results:
            print("Starting preprocessing data")
            train_input = idx.convert_from_file(train_images)
            train = np.array([np.zeros(784) for i in range(len(train_input))])
            result_input = idx.convert_from_file(train_results)
            result = np.array([[int(i == result_input[j]) for i in range(10)] for j in range(len(train_input))])
            for j in range(len(train)):
                train[j] = train_input[j].reshape(1, 784)/255
            #print("Half way preprocessing data")
            #train = traite_entrees(train)
            print("Ending preprocessing data")
            return (train_input, train, result_input, result)

def MNIST_stoch_training(train_input, train, result_input, result, reseau, nbr):
    print("Starting generating weight and bias")
    (W, B) = random_w_b(train[0], reseau)
    print("Ending generating weight and bias")
    print("Starting training neural network")
    (nW, nB, E) = stochastic_training(train[:nbr], result[:nbr], W, B, 0.01, reseau, iterations=10)
    print("Ending training neural network")
    return (nW, nB, E)

def Global_MNIST(nbr):
    reseau = [16, 16, 10]
    (train_input, train, result_input, result) = MNIST_datas()
    (W, B, E) = MNIST_stoch_training(train_input, train, result_input, result, reseau, nbr)
    success = np.array([0, 0])
    for i in range(20000):
        res = front_prop(train[nbr+i], reseau, W, B)
        if res[-1][result_input[nbr+i]] == np.max(res[-1]):
            success[1] += 1
        success[0] += 1
    print("success rate:  " + str(success[1]/success[0]))
    EE = [[] for i in range(10)]
    for i, ei in enumerate(E):
        EE[result_input[i]].append(ei)
    # for i in range(len(E)) :
    #     EE[result_input[i]].append(E[i])
    # for i in range(10):
    #     plt.plot(EE[i])
    # plt.show()
    save_network(reseau, W, B, "test.data")
    return (W, B, EE)

# (train_input, train, result_input, result) = MNIST_datas()
# Global_MNIST(40000)


def image(k=-1): #Affiche les iamges de MNIST pour peu qu'on ai lancé datas avant
    if k == -1:
        k = rd.randint(0, 60000-1)
    res = np.array([[[float((1-train[k][j+28*i])*255) for ii in range(3)] for j in range(len(train_input[k]))] for i in range(len(train_input[k]))])
    plt.imshow(res)

# image()
# plt.show()
                    