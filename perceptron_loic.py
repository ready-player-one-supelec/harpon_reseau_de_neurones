# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:13:36 2018

@author: Loic
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd

def sigmoid(x):
    return  1/(1+np.exp(-1*x)) #On choisit cette sigmoide car la dérivée est simple

def couche(inputs,weight,bias):
    prod = np.dot(weight,inputs) + bias
    res = sigmoid(prod)
    return res

def reseau(inputs,total_weight,total_bias):
    n = len(total_weight)
    res = [inputs]
    for i in range(n):
        ouptut = couche(res[i],total_weight[i],total_bias[i])
        res.append(ouptut)
    return res

def erreur(inputs,total_weight,total_bias,th_output):#Inutile dans le programme actuel
    ouput = reseau(inputs,total_weight,total_bias)[-1]
    res = 0 
    for i in range(len(ouput)):
        res  = res + (th_output[i]-ouput[i])**2
    return res

def gradient(inputs,total_weight,total_bias,th_output,output):
    n = len(total_weight) #nombres de matrices donc de couches
    total_grad_weight = [np.array([]) for i in range(n)] #On va ajouter les matrices au fur et à mesure
    total_grad_bias = [np.array([]) for i in range(n)]
    total_grad = [np.array([]) for i in range(n+1)] #Sert à ne pas recalculer les gradients globaux à chause fois 
    total_grad[n] = 2*(output[n]-th_output)
    for i in range(n-1,-1,-1): #On part de la dérnère couche vers la première
        m = len(total_weight[i]) #Les matrices n'ont pas toutes la meme taille suivant le nombre de neurones 
        p = len(total_weight[i][0])  #Maintenant que c'est une matrice p est constant 
        total_grad_weight[i] = np.array([[0.0 for ii in range(p)] for jj in range(m)]) #Matrice des gradients des poids 
        total_grad[i] =  np.array([[0.0] for jj in range(p)])
        total_grad_bias[i] =  np.array([[0.0] for jj in range(m)])
        for j in range(m):
            summ = 0 #Somme les gradients de la couche precedente pour calculer les suivants
            for ii in range(len(total_grad[i+1])):
                summ += total_grad[i+1][ii][0]
            total_grad_bias[i][j] += summ* output[i+1][j] * (1 - output[i+1][j])
            for k in range(p):
                grad_base_weight = output[i][k] * output[i+1][j] * (1 - output[i+1][j]) #Comme output[0] = input on doit ajouter 1 à i pour avoir le bon résultat
                grad_base = total_weight[i][j][k] * output[i+1][j] * (1 - output[i+1][j])
                total_grad[i][k] += summ*grad_base
                total_grad_weight[i][j][k] += summ*grad_base_weight
    return (total_grad_weight,total_grad_bias)
  

def Stochastic(total_inputs,total_ouputs,ini_weight,ini_bias,vitesse):
    n = len(total_inputs)
    W = ini_weight
    B = ini_bias
    for i in range(n):
        I = total_inputs[i]
        O = total_ouputs[i]
        R = reseau(I,W,B)
        (gW,gB) = gradient(I,W,B,O,R)
        W = [W[i] - vitesse*gW[i] for i in range(len(W))]
        B = [B[i] - vitesse*gB[i] for i in range(len(B))]
    return (W,B)

def traite_entrees(total_inputs):
    n = len(total_inputs)
    m = len(total_inputs[0])
    moy = np.array([[0] for i in range(m)])
    for i in range(n):
        moy = moy + total_inputs[i]
    res = [total_inputs[i] - moy/n for i in range(n)]
    for i in range(n):
        norm = 0
        for j in range(m):
            norm += (res[i][j][0])**2
        norm = np.sqrt(norm)
        for j in range(m):
            res[i][j][0] = res[i][j][0]/norm
    return res

def initialise_weight(Ln):
      return 0
              
W = [np.array([[0.1,-0.8],[-0.3,0.6]]),np.array([[0.2,0.7]])]
B = [np.array([[0.7],[-0.7]]),np.array([0.8])]
I = [ np.array([[int(i%4 == 0 or i%4 == 1)],[int(i%4 == 0 or i%4 == 3)]]) for i in range(2048)]
J = traite_entrees(I)
O = [ np.array([int(i%4 == 1 or i%4 == 3)]) for i in range(2048)]
(nW,nB)=Stochastic(J,O,W,B,1)
R11 = reseau(I[0],nW,nB)
R10 = reseau(I[1],nW,nB)
R00 = reseau(I[2],nW,nB)
R01 = reseau(I[3],nW,nB)
print("11: " + str(R11[-1][0]))
print("10: " + str(R10[-1][0]))
print("01: " + str(R01[-1][0]))
print("00: " + str(R00[-1][0]))

def xor(a,b):
    res = reseau([[a],[b]],nW,nB)[-1][0]
    if res > .5 :
        return 1 
    else :
        return 0



