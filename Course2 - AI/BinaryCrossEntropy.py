import numpy as np
from sklearn.datasets import load_iris 

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def bce(y, y_hat):
    return - np.mean(y*np.log(y_hat) - (1 - y)*np.log(1 - y_hat))


class Perceptron():
  def __init__(self, size, activation, loss):
    self.w = np.random.randn(size) 
    self.ws = []
    self.activation = activation
    self.loss = loss
    
  def __call__(self, w, x):
    return self.activation(np.dot(x, w)) 

  def fit(self, x, y, epochs, lr):
    x = np.c_[np.ones(len(x)), x]
    for epoch in range(epochs):
        # Batch Gradient Descent
        y_hat = self(self.w, x)  
        # función de pérdida 
        l = self.loss(y, y_hat)
        # derivadas
        dldh = (y_hat - y)
        dhdw = x
        dldw = np.dot(dldh, dhdw)
        # actualizar pesos
        self.w = self.w - lr*dldw
        # guardar pesos para animación
        self.ws.append(self.w.copy())

iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
y = (iris.target == 0).astype(int)


np.random.seed(42)

perceptron = Perceptron(3, sigmoid, bce)
epochs, lr = 20, 0.01
perceptron.fit(X, y, epochs, lr)

w = perceptron.ws[-1]
x_new = [1, 2, 0.5]
y_pred = perceptron(w, x_new)
print(y_pred)
