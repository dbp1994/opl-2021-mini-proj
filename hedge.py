import numpy as np
import scipy
import matplotlib.pyplot as plt
import math 
from scipy.linalg import hadamard

A_d = hadamard(64, dtype='float')

k = 2 # 'good' experts
eps = 0.025 # 'good' experts are better than the rest by eps
T = 32768 # no. of rounds
print(A_d)

## preprocessing as mentioned in "Parameter-Free Hedging Algorithm (2009)"

const_lst = []
for i in range(A_d.shape[0]):
  if (np.sum(A_d[i,:]) == 1.*A_d.shape[1]) or (np.sum(A_d[i,:]) == (-1.)* A_d.shape[1]):
    const_lst.append(i)

A_d_del = np.delete(A_d, np.asarray(const_lst),axis=0)
# A_d_del.shape
A_d_neg = -A_d_del
A_d_stack = np.vstack((A_d_del, A_d_neg))

k_lst = np.arange(k) # np.random.choice(A_d_stack.shape[0],k, replace=False)
# k_lst
# A_d_stack[k_lst,:].shape

A_d_penul = np.tile(A_d_stack,int(T/A_d.shape[1]))
print(A_d_penul.shape) ## (126, 32768)


A_d_penul[:,0] /= 2
# print(A_d_penul[:,0])

dum = np.zeros_like(A_d_penul)
# dum.shape
dum[k_lst,:] = np.ones_like(A_d_penul[k_lst,:])
dum = np.multiply(dum, 0.025, casting='unsafe')
# dum[k_lst,:]

A_d_fin = A_d_penul - dum
# A_d_fin[k_lst,:]

# ===============================
# ===============================


## Hedge (eta = U/\sqrt(t)) - Algorithm

A_d_fin = np.clip(A_d_fin, 0, 1)

N = A_d_fin.shape[0]
U = 10.75

R = np.zeros((N,))
weight_vec = np.ones((N, ))
pred_prob = np.zeros((N, ))

for t in range(1, T+1):

  weight_sum = np.sum(weight_vec)

  if weight_sum <= 1e-10:
    weight_vec *= 1e10
    weight_sum *= 1e10
    # pred_prob = np.asarray([1./N]*N)
  
  pred_prob = (1/weight_sum)*weight_vec

  
  if t % 3000 == 0:
    print("\n======\n")
    print(pred_prob)
    print("\n======\n")
    print(1/np.sum(weight_vec))
    print("\n======\n")

  if (pred_prob == np.zeros((N,))).all():
    l = np.random.choice(N, 1)
    pred_prob = np.zeros((N, ))
    pred_prob[l] = 1

  if t > 1:
    # for i in range(N):
    weight_vec *= np.exp((-U/np.sqrt(t-1))*A_d_fin[:,t-1])

  loss_t = np.inner(A_d_fin[:,t-1], pred_prob) 
  temp_reward = (loss_t*np.ones((N,))) - A_d_fin[:,t-1]
  R += temp_reward

print(np.argsort(pred_prob))
print(f"Regret to best expert: {R[np.argsort(pred_prob)[-1]]}")