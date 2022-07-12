# Evalutate.py
import numpy as np
from sklearn.metrics import confusion_matrix

# Quadratic Weighted Kappa Metric
def qwk(y_true, y_pred, N=5):
    w = np.zeros((N,N))
    for i in range(len(w)): 
        for j in range(len(w)):
            w[i][j] = float(((i-j)**2)/(N-1)**2)

    O = confusion_matrix(y_true, y_pred)
    O = O / O.sum()

    act_hist=np.zeros([N])
    for i in y_true: 
        act_hist[i]+=1

    pred_hist=np.zeros([N])
    for i in y_pred:
        pred_hist[i]+=1

    E = np.outer(act_hist, pred_hist)
    
    num=0
    den=0
    for i in range(len(w)):
        for j in range(len(w)):
            num+=w[i][j]*O[i][j]
            den+=w[i][j]*E[i][j]
            
    return (1 - (num/den))