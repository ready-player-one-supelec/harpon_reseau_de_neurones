# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:10:09 2018

@author: Loic
"""

import numpy as np
import matplotlib.pyplot as plt
import idx2numpy as idx

#%% Neural Network Base

def sigmoid(x):
    return  1/(1+np.exp(-x)) #On choisit cette sigmoide car la dérivée est simple

def dsigmoid(m): #Work in progress do not use
    return m*(np.ones(len(m)) - m)

def front_prop(inputs,reseau,weights,bias,activation = sigmoid):
    #weights is a list of arrays (lines:layers[k],colons:layers[k+1]), bias is a list of arrays 
    #returns the list of results of perceptrons
    #reseau n inclut pas le layer d'entree
    #les weights et biais sont comptés avec leurs colonnes de sortie
    #list_out inclue l'entrée donc len(list_out)=len(reseau)+1
    list_out=[np.zeros(reseau[k]) for k in range(len(reseau))]
    list_out=[np.array(inputs)]+list_out
    for col in range(len(reseau)):
        sig_in=np.dot(list_out[col],weights[col])+bias[col]
        for per in range(len(sig_in)):
            list_out[col+1][per]=activation(sig_in[per])
    return list_out


def backprop(inputs,th_output,reseau,weights,bias,derivee = dsigmoid):
    #backpropagation:dc/daijk=dc/dbjk *Yik-1 et dc/bik=SUMj[dc/dbjk+1*aijk+1] *Yik+1(1-Yik+1)
    #matriciellement DAk=DBk.Yk-1 et DBk=Ak+1 . DBk+1*Yk+1 *(1-Yk+1)
    list_out=front_prop(inputs,reseau,weights,bias)    
    grad_weight=[weights[k]*0 for k in range(len(weights))]
    grad_bias=[bias[k]*0 for k in range(len(bias))]
    #init
    grad_bias[-1]= derivee(list_out[-1])*(list_out[-1]-np.array(th_output))
    grad_weight[-1]=np.dot(list_out[-2][None].T,grad_bias[-1][None])
    #recurrence
    for col in range(len(list_out)-3,-1,-1):
        grad_bias[col]=np.dot(weights[col+1],grad_bias[col+1])*derivee(list_out[col+1])
        grad_weight[col]=np.dot(list_out[col][None].T,grad_bias[col][None])
    return grad_weight,grad_bias,np.linalg.norm(th_output-list_out[-1])/2

#%% Batch Training
  
def random_w_b(inputs,reseau):
    weights=[2*np.random.random((len(inputs),reseau[0]))-np.ones((len(inputs),reseau[0]))]+[2*np.random.random((reseau[k],reseau[k+1]))-np.ones((reseau[k],reseau[k+1])) for k in range(len(reseau)-1)]
    bias=[np.zeros(reseau[k]) for k in range(len(reseau))]
    return weights, bias   

def batch_training(L_inputs,L_th_output,reseau,weights,bias,rate,iterations): 
    error = []
    for k in range(iterations):
        delta_weight = [weights[k]*0 for k in range(len(weights))]
        delta_bias = [bias[k]*0 for k in range(len(bias))]
        cost_tot = 0
        for data in range(len(L_inputs)):
            gw,gb,cost = backprop(L_inputs[data],L_th_output[data],reseau,weights,bias)
            for col in range(len(gw)):
                delta_weight[col] += gw[col]*rate/len(L_inputs)
                delta_bias[col] += gb[col]*rate/len(L_inputs)
            cost_tot += cost/len(L_inputs)
        error.append(cost_tot)
        for col in range(len(weights)):
            weights[col] += -delta_weight[col]
            bias[col] += -delta_bias[col]  
    return weights,bias,error


#%% Stochastic learning

def stochastic_training(total_inputs,total_ouputs,ini_weight,ini_bias,vitesse,reseau):
    n = len(total_inputs)
    W = ini_weight
    B = ini_bias
    E = []
    for i in range(n):
        I = total_inputs[i]
        O = total_ouputs[i]
        (gW,gB,ee) = backprop(I,O,reseau,W,B)
        E.append(ee)
        W = [W[i] - vitesse*gW[i] for i in range(len(W))]
        B = [B[i] - vitesse*gB[i] for i in range(len(B))]
        if i/n*100%1 == 0 :
            print(str(i/n*100) + " % done")
    return (W,B,E)


def traite_entrees(total_inputs): #Work in progress do not use
    n = len(total_inputs)
    m = len(total_inputs[0])
    moy = np.array([0 for i in range(m)])
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


#%% Xor
   
def le_xor_batch(pas,reseau = [4,1]):
    (Wt,Bt) = random_w_b([0,0],reseau)   
    N =10000
    I = [ np.array([int(i%4 == 0 or i%4 == 1),int(i%4 == 0 or i%4 == 3)]) for i in range(4)]
    O = [ np.array([int( (I[i][0] == 1 and I[i][1] == 0) or (I[i][0] == 0 and I[i][1] == 1) )]) for i in range(4)]
    (nW,nB,E)= batch_training(I,O,reseau,Wt,Bt,pas,N)
    R11 = front_prop(I[0],reseau,nW,nB)
    R10 = front_prop(I[1],reseau,nW,nB)
    R00 = front_prop(I[2],reseau,nW,nB)
    R01 = front_prop(I[3],reseau,nW,nB)
    plt.plot(E)
    print("11: " + str(R11[-1]) + " ie. " + str(int(R11[-1][0] > .5)))
    print("10: " + str(R10[-1]) + " ie. " + str(int(R10[-1][0] > .5)))
    print("01: " + str(R01[-1]) + " ie. " + str(int(R01[-1][0] > .5)))
    print("00: " + str(R00[-1]) + " ie. " + str(int(R00[-1][0] > .5)))


def le_xor_stochastic(pas,N = 2000, reseau = [4,1]):
    (Wt,Bt) = random_w_b([0,0],reseau)   
    I = [ np.array([int(i%4 == 0 or i%4 == 1),int(i%4 == 0 or i%4 == 3)]) for i in range(N)]
    O = [ np.array([int( (I[i][0] == 1 and I[i][1] == 0) or (I[i][0] == 0 and I[i][1] == 1) )]) for i in range(N)]
    (nW,nB,E)= stochastic_training(I,O,Wt,Bt,pas,reseau)
    R11 = front_prop(I[0],reseau,nW,nB)
    R10 = front_prop(I[1],reseau,nW,nB)
    R00 = front_prop(I[2],reseau,nW,nB)
    R01 = front_prop(I[3],reseau,nW,nB)
    plt.plot(E[::4])
    plt.plot(E[1::4])
    plt.plot(E[2::4])
    plt.plot(E[3::4])
    print("11: " + str(R11[-1]) + " ie. " + str(int(R11[-1][0] > .5)))
    print("10: " + str(R10[-1]) + " ie. " + str(int(R10[-1][0] > .5)))
    print("01: " + str(R01[-1]) + " ie. " + str(int(R01[-1][0] > .5)))
    print("00: " + str(R00[-1]) + " ie. " + str(int(R00[-1][0] > .5)))

#%% MNIST

def MNIST(nbr):
    with open(r"C:\Users\Loic\Documents\Projet Long HARPON\train-images.idx3-ubyte","rb") as train_images:
        with open(r"C:\Users\Loic\Documents\Projet Long HARPON\train-labels.idx1-ubyte","rb") as train_results:
            print("Strating preprocessing data")
            train_input = idx.convert_from_file(train_images)
            train = np.array([np.zeros(784).reshape(784,1) for i in range(len(train_input))])
            result_input = idx.convert_from_file(train_results)
            result = np.array([[[int(i == result_input[j])] for i in range(10)] for j in range(len(train_input))])
            for j in range(len(train)):
                train[j] = train_input[j].reshape(28*28,1)/255
            #print("Half way preprocessing data")
            #train = traite_entrees(train)
            print("Ending preprocessing data")
            print("Strating generating weight and bias")
            (W,B) = initialise_weight_bias_mnist([784,16,16,10])
            print("Ending generating weight and bias")
            print("Starting training neural network")
            (nW,nB,E)=Stochastic(train[:nbr],result[:nbr],W,B,.05)
            print("Ending training neural network")
            success = np.array([0,0])
            for i in range(20000):
                res = reseau(train[nbr+i],nW,nB)
                if res[-1][result_input[nbr+i]] == np.max(res[-1]):
                     success[1] +=  1
                success[0] += 1
            print("success rate:  " + str(success[1]/success[0]))
            EE = [[] for i in range(10)]
            for i in range(len(E)) :
                EE[result_input[i]].append(E[i])
            for i in range(10):
                plt.plot(EE[i])
            return (nW,nB,EE)


#(W,B,E) = MNIST(30000)
#(w,b,ee) = MNIST(5000)


             
def datas(): #Fonction de debugging inutile dans un processus normal
     with open(r"C:\Users\Loic\Documents\Projet Long HARPON\train-images.idx3-ubyte","rb") as train_images:
        with open(r"C:\Users\Loic\Documents\Projet Long HARPON\train-labels.idx1-ubyte","rb") as train_results:
            print("Strating preprocessing data")
            train_input = idx.convert_from_file(train_images)
            train = np.array([np.zeros(784).reshape(784,1) for i in range(len(train_input))])
            result_input = idx.convert_from_file(train_results)
            result = np.array([[[int(i == result_input[j])] for i in range(10)] for j in range(len(train_input))])
            for j in range(len(train)):
                train[j] = train_input[j].reshape(28*28,1)/255
            print("Half way preprocessing data")
            train = traite_entrees(train)
            print("Ending preprocessing data")
            return (train_input,train,result,result_input) 

#(train_input,train,result,result_input) = datas()

def image(k=-1): #Affiche les iamges de MNIST pour peu qu'on ai lancé datas avant 
   if k == -1 :
       k = rd.randint(0,60000-1)
   res = np.array([[[int((1-train[k][j+28*i])*230) for ii in range(3)] for j in range(len(train_input[k]))] for i in range(len(train_input[k]))])
   plt.imshow(res)      
                    