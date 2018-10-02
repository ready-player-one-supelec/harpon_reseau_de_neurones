import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def front_prop(Xin,list_size_layers,weights,bias):
    #weights is a list of arrays (lines:layers[k],colons:layers[k+1]), bias is a list of arrays 
    #returns the list of results of perceptrons
    #list_size_layers n inclut pas le layer d'entree
    #les weights et biais sont comptés avec leurs colonnes de sortie
    #list_out inclue l'entrée donc len(list_out)=len(list_size_layers)+1
    list_out=[np.zeros(list_size_layers[k]) for k in range(len(list_size_layers))]
    list_out=[np.array(Xin)]+list_out
    for col in range(len(list_size_layers)):
        sig_in=np.dot(list_out[col],weights[col])+bias[col]
        for per in range(len(sig_in)):
            list_out[col+1][per]=sigmoid(sig_in[per])
    return list_out
     
def backprob(Xin,Yth,list_size_layers,weights,bias):
    #backpropagation:dc/daijk=dc/dbjk *Yik-1 et dc/bik=sum dc/dbjk+1*aijk+1*Yjk+1(1-Yjk+1)
    #matriciellement DAk=DBk.Yk-1 et DBk=Ak+1 . DBk+1*Yk+1 *(1-Yk+1)
    list_out=front_prop(Xin,list_size_layers,weights,bias)    
    grad_weight=[weights[k]*0 for k in range(len(weights))]
    grad_bias=[bias[k]*0 for k in range(len(bias))]
    #init
    grad_bias[-1]=list_out[-1]*(np.array(Yth)-list_out[-1])*(np.ones(len(list_out[-1]))-list_out[-1])
    grad_weight[-1]=np.dot(list_out[-2][None].T,np.array([grad_bias[-1]]))
    #recurrence
    for col in range(len(list_out)-3,-1,-1):
        grad_bias[col]=np.dot(weights[col+1],grad_bias[col+1]*list_out[col+1]*(np.ones(len(list_out[col+1])))-list_out[col+1])
        grad_weight[col]=np.dot(list_out[col][None].T,np.array([grad_bias[col]]))
    return grad_weight,grad_bias


def training(listXin,listYth,list_size_layers,weights,bias,step,iterations):
    for k in range(len(iterations)):
        delta_weight=[weights[k]*0 for k in range(len(weights))]
        delta_bias=[bias[k]*0 for k in range(len(bias))]
        for data in range(len(listXin)):
            gw,gb=backprob(listXin[data],listYth[data],list_size_layers,weights,bias)
            for col in range(len(gw)):
                delta_weight[col]+=gw[col]*step/len(gw)
                delta_bias[col]+=gb[col]*step/len(gw)
        weights+=-delta_weight
        bias+=-delta_bias
        return weights,bias


     