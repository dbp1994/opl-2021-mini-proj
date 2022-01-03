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

## AdaNormalHedge - Algorithm

N = A_d_penul.shape[0]
prior = np.asarray([1./N]*N)
R = np.zeros((N,))
C = np.zeros((N,))
weight_vec = np.zeros((N,))
pred_prob = np.zeros((N,))

def Phi_pot(R, C):
  if R == 0 and C == 0:
    return 1
  else:
    return np.exp(((max(0., R))**2)/(3*C))

def w(R_t,C_t): 
  return 0.5*(Phi_pot(R_t + 1,C_t + 1) - Phi_pot(R_t - 1,C_t + 1))

for t in range(T):
  for i in range(N):
    weight_vec[i] = w(R[i], C[i])

  # norm_const = np.inner(weight_vec, prior)
  if t % 3000 == 0:
    print("\n======\n")
    print(weight_vec)
    print("\n======\n")
  pred_prob = (1/np.sum(weight_vec))*weight_vec
  # if (pred_prob == np.zeros((N,))).all():
  #   l = np.random.choice(N, 1)
  #   pred_prob = np.zeros((N, ))
  #   pred_prob[l] = 1
  loss_t = np.inner(A_d_fin[:,t], pred_prob) 
  temp_reg = (loss_t*np.ones((N,))) - A_d_fin[:,t]
  R += temp_reg
  C += np.absolute(temp_reg)


print(np.argsort(pred_prob))
print(f"Regret to best expert: {R[np.argsort(pred_prob)[-1]]}")