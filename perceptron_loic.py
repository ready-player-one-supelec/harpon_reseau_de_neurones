# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:13:36 2018

@author: Loic
"""
import numpy as np

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

def gradient(inputs,total_weight,total_bias,th_output,outpup):
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
  


              
W = [np.array([[0.5,0.5],[0.5,0.5]]),np.array([[0.5,0.5]])]
B = [np.array([[0],[0]]),np.array([0])]
I = np.array([[1],[0]])
O = np.array([1])

R = reseau(I,W,B)
(gW,gB) = gradient(I,W,B,O,R)
nW = [W[i] - gW[i] for i in range(len(W))]
nB = [B[i] - gB[i] for i in range(len(B))]
E = erreur(I,W,B,O)
nE = erreur(I,nW,nB,O)
nR = reseau(I,nW,nB)



