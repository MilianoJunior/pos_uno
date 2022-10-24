# -*- coding: utf-8 -*-
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Perception:
  def __init__(self, eta=0.01, n_iter=10) -> None:
    self.eta = eta
    self.n_iter = n_iter

  def fit(self, X, y):
    self.w_ = np.zeros(1 + X.shape[0])
    self.errors_ = []

    for _ in range(self.n_iter):
      errors = 0
      for xi, target in zip(X,y):
        previsao = self.predict(xi)
        print(previsao)
        update = self.eta * (target - previsao)
        print('ref: ', xi, 'previsao: ',previsao,'tagert: ',target,'Erro: ',update)
        self.w_[1:] += update * xi
        self.w_[0] += update
        errors += int(update != 0.0)
      self.errors_.append(errors)

  def predict(self, X):
      op = np.where(self.net_input(X) >= 0.0, 1, -1)
      # print('predict',X, op)
      return op

  def net_input(self, X):
      cp = np.dot(X, self.w_[1]) + self.w_[0]
      # print('net_input',X, cp)
      return cp


def dados(input_a):
  if input_a > 5:
    return 1
  else:
    return -1

X = []
y = []
for a in range(0,10):
  X.append(a)
  y.append(dados(a))
X = np.array(X)
y = np.array(y)
print(X)
print(y)

rede = Perception(eta=0.01, n_iter=20)
rede.fit(X,y)