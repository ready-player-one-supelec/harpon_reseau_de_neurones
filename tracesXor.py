# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:26:12 2018

@author: sacha
"""
import numpy as np
import matplotlib.pyplot as plt


#%%batch
def plot_xor_batch(reseau, iterations, points, incer=1000):
    errors = np.zeros((iterations, points))
    I = [
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 1])
    ]
    O = [np.array([0]), np.array([1]), np.array([1]), np.array([0])]
    for k in range(iterations):
        print(k)
        W, B = random_w_b(I[0], reseau)
        W, B, E = batch_training(I, O, reseau, W, B, 0.35, points)
        errors[k] = E
    Mean = np.average(errors, 0)
    STD = np.std(errors, 0)
    plt.plot(
        Mean,
        'b',
        label='erreur moyenne sur ' + str(iterations) +
        ' runs par itération ; réseau 2-4-1 en Batch learning; sigma=1/(1+e(-x))'
    )
    for k in range(points // incer):
        plt.plot([incer * k, incer * k], [
            Mean[incer * k] - STD[incer * k] / 2,
            Mean[incer * k] + STD[incer * k] / 2
        ], 'b*-')
    plt.legend()
    plt.plot([0, 0.001], [0, 0])
    plt.grid()
    return None


def plot_xor_batch_tanh(reseau, iterations, points, incer=1000, color='b'):
    errors = np.zeros((iterations, points))
    I = [
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 1])
    ]
    I = traite_entrees(I)
    O = [np.array([0]), np.array([1]), np.array([1]), np.array([0])]
    for k in range(iterations):
        print(k)
        W, B = random_w_b(I[0], reseau)
        W, B, E = batch_training(I, O, reseau, W, B, 0.035, points, tanh,
                                 dtanh)
        errors[k] = E
    Mean = np.average(errors, 0)
    STD = np.std(errors, 0)
    plt.plot(
        Mean,
        color,
        label='erreur moyenne sur ' + str(iterations) +
        ' runs par itération pour un réseau 2, ' + str(reseau)[1:-1] +
        ' en Batch learning avec sigma=tanh(x); entrees traitees')
    for k in range(points // incer):
        plt.plot([incer * k, incer * k], [
            Mean[incer * k] - STD[incer * k] / 2,
            Mean[incer * k] + STD[incer * k] / 2
        ], color + '*-')
    plt.legend()
    plt.grid()
    return None


#%% stochastique


def plot_xor_stoch(reseau, iterations, points, incer=100):
    errors = np.zeros((iterations, points))
    I = [
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 1])
    ]
    O = [np.array([0]), np.array([1]), np.array([1]), np.array([0])]
    for k in range(iterations):
        print(k)
        W, B = random_w_b(I[0], reseau)
        for i in range(points // 4):
            W, B, useless = stochastic_training(I, O, W, B, 0.35, reseau, 1)
            errors[k][4 * i] = cost_function(I, O, reseau, W, B)
            errors[k][4 * i + 1] = errors[k][4 * i]
            errors[k][4 * i + 2] = errors[k][4 * i]
            errors[k][4 * i + 3] = errors[k][4 * i]
    Mean = np.average(errors, 0)
    STD = np.std(errors, 0)
    plt.plot(
        Mean,
        'r',
        label='erreur moyenne sur ' + str(iterations) +
        ' runs par itération ; réseau 2-4-1 en stochastic learning; sigma=1/(1+e(-x))'
    )
    for k in range(points // incer):
        plt.plot([incer * k, incer * k], [
            Mean[incer * k] - STD[incer * k] / 2,
            Mean[incer * k] + STD[incer * k] / 2
        ], 'r*-')
    plt.legend()
    plt.grid()
    return None


def plot_xor_stoch_tanh(reseau, iterations, points, incer=100):
    errors = np.zeros((iterations, points))
    I = [
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 1])
    ]
    I = traite_entrees(I)
    O = [np.array([0]), np.array([1]), np.array([1]), np.array([0])]
    for k in range(iterations):
        print(k)
        W, B = random_w_b(I[0], reseau)
        for i in range(points // 4):
            W, B, useless = stochastic_training(I, O, W, B, 0.035, reseau, 1,
                                                tanh, dtanh)
            errors[k][4 * i] = cost_function(I, O, reseau, W, B, tanh)
            errors[k][4 * i + 1] = errors[k][4 * i]
            errors[k][4 * i + 2] = errors[k][4 * i]
            errors[k][4 * i + 3] = errors[k][4 * i]
    Mean = np.average(errors, 0)
    STD = np.std(errors, 0)
    plt.plot(
        Mean,
        'r',
        label='erreur moyenne sur ' + str(iterations) +
        ' runs par itération ; réseau 2-4-1 en stochastic learning; sigma=tanh(x); entrees traitees'
    )
    for k in range(points // incer):
        plt.plot([incer * k, incer * k], [
            Mean[incer * k] - STD[incer * k] / 2,
            Mean[incer * k] + STD[incer * k] / 2
        ], 'r*-')
    plt.legend()
    plt.grid()
    return None
