import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

x = np.random.rand(20)
y = 2*x + (np.random.rand(20)-0.5)*0.5

def gradient(w, x, y):
    dldw = x*w - y
    dydw = x
    dldw = dldw*dydw
    return np.mean(2*dldw)

def cost(y, y_hat):
    return ((y_hat - y)**2).mean()

def solve(epochs=29, w=1.2, lr=0.2):
    weights = [(w, gradient(w, x, y), cost(x*w, y))]
    for i in range(1, epochs+1):
        dw = gradient(w, x, y)
        w = w - lr*dw
        weights.append((w, dw, cost(x*w, y)))
    return weights

# Obtener pesos y pérdidas durante el descenso de gradiente
weights = solve()

# Graficar la evolución del descenso de gradiente
epochs = range(len(weights))
losses = [weight[2] for weight in weights]

plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, label='Función de Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Evolución del Descenso de Gradiente')
plt.legend()
plt.grid(True)
plt.show()

# Ajustar regresión lineal usando los pesos finales
w_final = weights[-1][0]
regression_line = w_final * x

# Graficar puntos aleatorios y la línea de regresión lineal ajustada
plt.figure(figsize=(10, 5))
plt.scatter(x, y, label='Puntos Aleatorios')
plt.plot(x, regression_line, color='red', label='Regresión Lineal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste de Regresión Lineal a Puntos Aleatorios')
plt.legend()
plt.grid(True)
plt.show()

