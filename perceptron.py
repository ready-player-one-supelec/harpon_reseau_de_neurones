# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:35:18 2018

@author: sacha
"""
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def front_prop(Xin,list_size_layers,weights,bias):
    #weights is a list of list of list (colon then out then in), bias is a list of list (colon then line)
    #returns the list of results of perceptrons
    #list_size_layers n inclut pas le layer d'entree
    #les weights et biais sont comptés avec leurs colonnes de sortie
    #list_out inclue l'entrée donc len(list_out)=len(list_size_layers)+1
    list_out=[[] for k in range(len(list_size_layers))]
    list_out=[Xin] + list_out
    for col in range (len(list_size_layers)):
        for perceptron in range (len(list_size_layers[col])):
            sigin=0
            for i in range(len(list_size_layers[col-1])):
                sigin+=list_out[col][i]*weights[col][perceptron][i]            
        list_out[col+1].append(sigmoid(sigin+bias[col][perceptron]))
    return list_out

     
def backprob(Xin,Yth,list_size_layers,weights,bias):
    list_out=front_prop(Xin,list_size_layers,weights,bias)
    grad_weight=[[[0 for k in range(weights[k][i])]for i in range(len(weights[k]))]  for k in range(len(weights))]
    grad_bias=[[0 for i in range(len(bias[k]))]  for k in range(len(bias))]
    #backpropagation:dc/daijk=dc/dbjk *Yik-1 et dc/bik=sum dc/dbjk+1*aijk+1*Yjk+1(1-Yjk+1)
    for perceptron in range(len(bias[-1])):#initialisation 
        grad_bias[-1][perceptron]=2*(list_out[-1][perceptron]-Yth[perceptron])*list_out[-1][perceptron]*(1-list_out[-1][perceptron])
        for perin in range(len(bias[-2])):
            grad_weight[-1][perceptron][perin]=grad_bias[-1][perceptron]*list_out[-2][perin]
    for col in range(len(list_out)-2,-1,-1):#k
        for perceptron in range(len(list_size_layers[col])-1):
            for nextper in range(len(list_size_layers)[col+1]):
                grad_bias[col][perceptron]+=grad_bias[col+1][nextper]*grad_weight[col+1][nextper][perceptron]*list_out[col+1][nextper]*(1-list_out[col+1][nextper])
            for prevper in range(len(list_size_layers[col-1])):
                grad_weight[col][perceptron][prevper]=grad_bias[col][perceptron]*list_out[col-1][prevper]
    return grad_weight,grad_bias



     