import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap


#MLP Y CAPAS
class MLP:
    def __init__(self, layers):
        # el MLP es una lista de capas
        self.layers = layers

    def __call__(self, x):
        # calculamos la salida del modelo aplicando
        # cada capa de manera secuencial
        for layer in self.layers:
            x = layer(x)
        return x
    
class Layer():
    def __init__(self):
        self.params = []
        self.grads = []

    def __call__(self, x):
        # por defecto, devolver los inputs
        # cada capa hará algo diferente aquí
        return x

    def backward(self, grad):
        # cada capa, calculará sus gradientes
        # y los devolverá para las capas siguientes
        return grad

    def update(self, params):
        # si hay parámetros, los actualizaremos
        # con lo que nos de el optimizer
        return

class Linear(Layer):
    def __init__(self, d_in, d_out):
        # pesos de la capa
        self.w = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2/(d_in+d_out)),
                                  size=(d_in, d_out))
        self.b = np.zeros(d_out)

    def __call__(self, x):
        self.x = x
        self.params = [self.w, self.b]
        # salida del preceptrón
        return np.dot(x, self.w) + self.b    
    
    def backward(self, grad_output):
        # gradientes para la capa siguiente (BACKPROP)
        grad = np.dot(grad_output, self.w.T)
        self.grad_w = np.dot(self.x.T, grad_output)
        # gradientes para actualizar pesos
        self.grad_b = grad_output.mean(axis=0)*self.x.shape[0]
        self.grads = [self.grad_w, self.grad_b]
        return grad

    def update(self, params):
        self.w, self.b = params

class ReLU(Layer):
    def __call__(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad = self.x > 0
        return grad_output*grad
    
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=-1,keepdims=True)

class Sigmoid(Layer):    
    def __call__(self, x):
        self.x = x
        return sigmoid(x)

    def backward(self, grad_output):
        grad = sigmoid(self.x)*(1 - sigmoid(self.x))
        return grad_output*grad
    
#OPTIMIZADOR: STOCHASTIC GRADIENT DESCENT
class SGD():
    def __init__(self, net, lr):
        self.net = net
        self.lr = lr

    def update(self):
        for layer in self.net.layers:
            layer.update([
                params - self.lr*grads
                for params, grads in zip(layer.params, layer.grads)
            ])


#FUNCIONES DE PERDIDA
class Loss():
    def __init__(self, net):
        self.net = net

    def backward(self):
        # derivada de la loss function con respecto 
        # a la salida del MLP
        grad = self.grad_loss()
        # BACKPROPAGATION
        for layer in reversed(self.net.layers):
            grad = layer.backward(grad)
            
class MSE(Loss):
    def __call__(self, output, target):
        self.output, self.target = output, target.reshape(output.shape)
        loss = np.mean((self.output - self.target)**2)
        return loss.mean()

    def grad_loss(self):
        return self.output -  self.target
    
class BCE(Loss):
    def __call__(self, output, target):
        self.output, self.target = output, target.reshape(output.shape)
        loss = - np.mean(self.target*np.log(self.output) - (1 - self.target)*np.log(1 - self.output))
        return loss.mean()

    def grad_loss(self):
        return self.output -  self.target
            
class CrossEntropy(Loss):
    def __call__(self, output, target):
        self.output, self.target = output, target
        logits = output[np.arange(len(output)), target]
        loss = - logits + np.log(np.sum(np.exp(output), axis=-1))
        loss = loss.mean()
        return loss

    def grad_loss(self):
        answers = np.zeros_like(self.output)
        answers[np.arange(len(self.output)), self.target] = 1
        return (- answers + softmax(self.output)) / self.output.shape[0]
    

#REGRESION
m = 100
X = 6 * np.random.rand(m, 1) - 3
Y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

D_in, H, D_out = 1, 3, 1
mlp = MLP([
    Linear(D_in, H),
    ReLU(),
    Linear(H, D_out)
])

optimizer = SGD(mlp, lr=0.01)
loss = MSE(mlp)

epochs = 100
batch_size = 10

batches = len(X) // batch_size
log_each = 10
l = []
for e in range(1,epochs+1):
    _l = []
    for b in range(batches):
        x = X[b*batch_size:(b+1)*batch_size]
        y = Y[b*batch_size:(b+1)*batch_size] 
        y_pred = mlp(x)    
        _l.append(loss(y_pred, y))
        loss.backward()    
        optimizer.update()
    l.append(np.mean(_l))
    if not e % log_each:
        print(f'Epoch {e}/{epochs}, Loss: {np.mean(l):.4f}')


x_new = np.linspace(-3, 3, 100)
x_new = x_new.reshape(len(x_new),1)
y_pred = mlp(x_new)
    
plt.plot(X, Y, "b.")
plt.plot(x_new, y_pred, "-k")
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$y$", rotation=0, fontsize=14)
plt.grid(True)
plt.show()

#CLASIFICACION BINARIA

iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
Y = iris.target

# normalización datos
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_norm = (X - X_mean) / X_std
Y = (iris.target == 0).astype(int)

D_in, H, D_out = 2, 3, 1
mlp = MLP([
    Linear(D_in, H),
    ReLU(),
    Linear(H, D_out),
    Sigmoid()
])

optimizer = SGD(mlp, lr=0.01)
loss = BCE(mlp)

epochs = 100
batch_size = 10

batches = len(X) // batch_size
log_each = 10
l = []
for e in range(1,epochs+1):
    _l = []
    for b in range(batches):
        x = X_norm[b*batch_size:(b+1)*batch_size]
        y = Y[b*batch_size:(b+1)*batch_size] 
        y_pred = mlp(x)    
        _l.append(loss(y_pred, y))
        loss.backward()    
        optimizer.update()
    l.append(np.mean(_l))
    if not e % log_each:
        print(f'Epoch {e}/{epochs}, Loss: {np.mean(l):.4f}')

custom_cmap = ListedColormap(['#9898ff', '#fafab0'])

axes = [0.5, 5, 0, 2]
label="Iris Setosa"
x0, x1 = np.meshgrid(
        np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
        np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]
X_new = (X_new - X_mean) / X_std 
y_predict = mlp(X_new)
zz = y_predict.reshape(x0.shape)
plt.plot(X[Y==0, 0], X[Y==0, 1], "bs", label=f"No {label}")
plt.plot(X[Y==1, 0], X[Y==1, 1], "yo", label=label)
plt.contourf(x0, x1, zz, cmap=custom_cmap)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="lower right", fontsize=14)
plt.axis(axes)
plt.show()

#CLASIFICACION MULTICLASE
Y = iris.target

D_in, H, D_out = 2, 100, 3
mlp = MLP([
    Linear(D_in, H),
    ReLU(),
    Linear(H, H),
    ReLU(),
    Linear(H, H),
    ReLU(),
    Linear(H, D_out)
])

optimizer = SGD(mlp, lr=0.02)
loss = CrossEntropy(mlp)

epochs = 100
batch_size = 10

batches = len(X) // batch_size
log_each = 10
l = []
for e in range(1,epochs+1):
    _l = []
    for b in range(batches):
        x = X_norm[b*batch_size:(b+1)*batch_size]
        y = Y[b*batch_size:(b+1)*batch_size] 
        y_pred = mlp(x)    
        _l.append(loss(y_pred, y))
        loss.backward()    
        optimizer.update()
    l.append(np.mean(_l))
    if not e % log_each:
        print(f'Epoch {e}/{epochs}, Loss: {np.mean(l):.4f}')


resolution=0.02
markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(Y))])

# plot the decision surface
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                       np.arange(x2_min, x2_max, resolution))
X_new = (np.array([xx1.ravel(), xx2.ravel()]).T - X_mean)/X_std
Z = mlp(X_new)
Z = np.argmax(softmax(Z), axis=1) 
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())
plt.xlabel('petal length', fontsize=14)
plt.ylabel('petal width', fontsize=14)
classes = ["Iris-Setosa", "Iris-Versicolor", "Iris-Virginica"]
for idx, cl in enumerate(np.unique(Y)):
    plt.scatter(x=X[Y == cl, 0], 
                y=X[Y == cl, 1],
                alpha=0.8, 
                c=colors[idx],
                marker=markers[idx], 
                label=classes[cl], 
                edgecolor='black')
plt.legend(loc='upper left', fontsize=14)
plt.show()