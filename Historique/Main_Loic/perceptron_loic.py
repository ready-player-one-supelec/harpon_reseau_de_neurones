# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:13:36 2018

@author: Loic
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import idx2numpy as idx

#%% Neural Network


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
    return res


def gradient(inputs, total_weight, total_bias, th_output, output):
    n = len(total_weight)  #nombres de matrices donc de couches
    total_grad_weight = [np.array([]) for i in range(n)
                         ]  #On va ajouter les matrices au fur et à mesure
    total_grad_bias = [np.array([]) for i in range(n)]
    total_grad = [
        np.array([]) for i in range(n + 1)
    ]  #Sert à ne pas recalculer les gradients globaux à chause fois
    total_grad[n] = 2 * (output[n] - th_output)
    for i in range(n - 1, -1,
                   -1):  #On part de la dérnère couche vers la première
        m = len(
            total_weight[i]
        )  #Les matrices n'ont pas toutes la meme taille suivant le nombre de neurones
        p = len(total_weight[i]
                [0])  #Maintenant que c'est une matrice p est constant
        total_grad_weight[i] = np.array(
            [[0.0 for ii in range(p)]
             for jj in range(m)])  #Matrice des gradients des poids
        total_grad[i] = np.array([[0.0] for jj in range(p)])
        total_grad_bias[i] = np.array([[0.0] for jj in range(m)])
        for j in range(m):
            summ = 0  #Somme les gradients de la couche precedente pour calculer les suivants
            for ii in range(len(total_grad[i + 1])):
                summ += total_grad[i + 1][ii][0]
            total_grad_bias[i][j] += summ * output[i + 1][j] * (
                1 - output[i + 1][j])
            for k in range(p):
                grad_base_weight = output[i][k] * output[i + 1][j] * (
                    1 - output[i + 1][j]
                )  #Comme output[0] = input on doit ajouter 1 à i pour avoir le bon résultat
                grad_base = total_weight[i][j][k] * output[i + 1][j] * (
                    1 - output[i + 1][j])
                total_grad[i][k] += summ * grad_base
                total_grad_weight[i][j][k] += summ * grad_base_weight
    return (total_grad_weight, total_grad_bias)


#%% Stochastic learning


def Stochastic(total_inputs, total_ouputs, ini_weight, ini_bias, vitesse):
    n = len(total_inputs)
    W = ini_weight
    B = ini_bias
    E = []
    for i in range(n):
        I = total_inputs[i]
        O = total_ouputs[i]
        R = reseau(I, W, B)
        E.append(erreur(O, R))
        (gW, gB) = gradient(I, W, B, O, R)
        W = [W[i] - vitesse * gW[i] for i in range(len(W))]
        B = [B[i] - vitesse * gB[i] for i in range(len(B))]
        if i / n * 1000 % 1 == 0:
            print(str(i / n * 100) + " % done")
    return (W, B, E)


def traite_entrees(total_inputs):
    n = len(total_inputs)
    m = len(total_inputs[0])
    moy = np.array([[0] for i in range(m)])
    for i in range(n):
        moy = moy + total_inputs[i]
    res = [total_inputs[i] - moy / n for i in range(n)]
    for i in range(n):
        norm = 0
        for j in range(m):
            norm += (res[i][j][0])**2
        norm = np.sqrt(norm)
        for j in range(m):
            res[i][j][0] = res[i][j][0] / norm
    return res


def initialise_weight_bias(Ln):
    res_w = []
    res_b = []
    for i in range(1, len(Ln)):
        mat_w = np.array([[(rd.random() - 0.5) * 2 for j in range(Ln[i - 1])]
                          for k in range(Ln[i])])
        res_w.append(mat_w)
        mat_b = np.array([[(rd.random() - 0.5) * 2] for k in range(Ln[i])])
        res_b.append(mat_b)
    return (res_w, res_b)


#%% Xor


def le_xor():
    (Wt, Bt) = initialise_weight_bias([2, 5, 1])
    N = 20000
    I = [
        np.array([[int(i % 4 == 0 or i % 4 == 1)],
                  [int(i % 4 == 0 or i % 4 == 3)]]) for i in range(N)
    ]
    J = traite_entrees(I)
    O = [
        np.array([
            int((I[i][0][0] == 1 and I[i][1][0] == 0)
                or (I[i][0][0] == 0 and I[i][1][0] == 1))
        ]) for i in range(N)
    ]
    (nW, nB, E) = Stochastic(J, O, Wt, Bt, 0.05)
    R11 = reseau(J[0], nW, nB)
    R10 = reseau(J[1], nW, nB)
    R00 = reseau(J[2], nW, nB)
    R01 = reseau(J[3], nW, nB)
    #plt.plot(E)
    plt.plot(E[::4])
    plt.plot(E[1::4])
    plt.plot(E[2::4])
    plt.plot(E[3::4])
    print("11: " + str(R11[-1][0]) + " ie. " + str(int(R11[-1][0][0] > .5)))
    print("10: " + str(R10[-1][0]) + " ie. " + str(int(R10[-1][0][0] > .5)))
    print("01: " + str(R01[-1][0]) + " ie. " + str(int(R01[-1][0][0] > .5)))
    print("00: " + str(R00[-1][0]) + " ie. " + str(int(R00[-1][0][0] > .5)))


#%% MNIST


def MNIST(nbr):
    with open(
            r"C:\Users\Loic\Documents\Projet Long HARPON\train-images.idx3-ubyte",
            "rb") as train_images:
        with open(
                r"C:\Users\Loic\Documents\Projet Long HARPON\train-labels.idx1-ubyte",
                "rb") as train_results:
            print("Strating preprocessing data")
            train_input = idx.convert_from_file(train_images)
            train = np.array([
                np.zeros(784).reshape(784, 1) for i in range(len(train_input))
            ])
            result_input = idx.convert_from_file(train_results)
            result = np.array([[[int(i == result_input[j])] for i in range(10)]
                               for j in range(len(train_input))])
            for j in range(len(train)):
                train[j] = train_input[j].reshape(28 * 28, 1) / 255
            #print("Half way preprocessing data")
            #train = traite_entrees(train)
            print("Ending preprocessing data")
            print("Strating generating weight and bias")
            (W, B) = initialise_weight_bias_mnist([784, 16, 16, 10])
            print("Ending generating weight and bias")
            print("Starting training neural network")
            (nW, nB, E) = Stochastic(train[:nbr], result[:nbr], W, B, .05)
            print("Ending training neural network")
            success = np.array([0, 0])
            for i in range(20000):
                res = reseau(train[nbr + i], nW, nB)
                if res[-1][result_input[nbr + i]] == np.max(res[-1]):
                    success[1] += 1
                success[0] += 1
            print("success rate:  " + str(success[1] / success[0]))
            EE = [[] for i in range(10)]
            for i in range(len(E)):
                EE[result_input[i]].append(E[i])
            for i in range(10):
                plt.plot(EE[i])
            return (nW, nB, EE)


#(W,B,E) = MNIST(30000)
#(w,b,ee) = MNIST(5000)


def datas():  #Fonction de debugging inutile dans un processus normal
    with open(
            r"C:\Users\Loic\Documents\Projet Long HARPON\train-images.idx3-ubyte",
            "rb") as train_images:
        with open(
                r"C:\Users\Loic\Documents\Projet Long HARPON\train-labels.idx1-ubyte",
                "rb") as train_results:
            print("Strating preprocessing data")
            train_input = idx.convert_from_file(train_images)
            train = np.array([
                np.zeros(784).reshape(784, 1) for i in range(len(train_input))
            ])
            result_input = idx.convert_from_file(train_results)
            result = np.array([[[int(i == result_input[j])] for i in range(10)]
                               for j in range(len(train_input))])
            for j in range(len(train)):
                train[j] = train_input[j].reshape(28 * 28, 1) / 255
            print("Half way preprocessing data")
            train = traite_entrees(train)
            print("Ending preprocessing data")
            return (train_input, train, result, result_input)


#(train_input,train,result,result_input) = datas()


def image(k=-1
          ):  #Affiche les iamges de MNIST pour peu qu'on ai lancé datas avant
    if k == -1:
        k = rd.randint(0, 60000 - 1)
    res = np.array(
        [[[int((1 - train[k][j + 28 * i]) * 230) for ii in range(3)]
          for j in range(len(train_input[k]))]
         for i in range(len(train_input[k]))])
    plt.imshow(res)
