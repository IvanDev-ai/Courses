from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

iris = load_iris()

class Perceptron():
    def __init__(self, size):
        self.w = np.random.randn(size) 
        self.ws = []
    
    def __call__(self, w, x):
        return np.dot(x, w) > 0 

    def fit(self, x, y, epochs, lr):
        x = np.c_[np.ones(len(x)), x]
        for epoch in range(epochs):
            # Batch Gradient Descent
            y_hat = self(self.w, x)  
            # función de pérdida (MSE)
            l = 0.5*(y_hat - y)**2
            # derivadas
            dldh = (y_hat - y)
            dhdw = x
            dldw = np.dot(dldh, dhdw)
            # actualizar pesos
            self.w = self.w - lr*dldw
            # guardar pesos para animación
            self.ws.append(self.w.copy())

X = iris.data[:, (2, 3)]  # petal length, petal width
y = iris.target
y = (iris.target == 0).astype(int)

# Normalización
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_norm = (X - X_mean) / X_std

np.random.seed(42)

perceptron = Perceptron(3)
epochs, lr = 20, 0.1
perceptron.fit(X_norm, y, epochs, lr)
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, autoscale_on=False)

def plot(i, axes=[0, 5, 0, 2], label="Iris Setosa"):
    ax.clear()
    w = perceptron.ws[i]
    tit = ax.set_title(f"Epoch {i+1}", fontsize=14)
    x0, x1 = np.meshgrid(
            np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
            np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
        )
    X_new = (np.c_[x0.ravel(), x1.ravel()] - X_mean) / X_std
    X_new = np.c_[np.ones(len(X_new)), X_new] 
    y_predict = perceptron(w, X_new)
    zz = y_predict.reshape(x0.shape)
    ax.plot(X[y==0, 0], X[y==0, 1], "bs", label=f"No {label}")
    ax.plot(X[y==1, 0], X[y==1, 1], "yo", label=label)
    ax.contourf(x0, x1, zz, cmap=plt.cm.RdYlBu, alpha=0.3)
    ax.set_xlabel("Petal length", fontsize=14)
    ax.set_ylabel("Petal width", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.axis(axes)
    return ax

anim = animation.FuncAnimation(fig, plot, frames=epochs, interval=200)
plt.show()
