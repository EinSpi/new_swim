import matplotlib.pyplot as plt
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
x=np.linspace(-2,2,200)
relu=np.maximum(x,0)
sig2=sigmoid(2*x)
sig3=sigmoid(3*x)
sig4=sigmoid(4*x)
sig5=sigmoid(5*x)
tanh=np.tanh(x)
sig=sigmoid(x)

plt.figure(figsize=(5, 5))
plt.plot(x,sig, label='a=1', color='red')
plt.plot(x,sig2, label='a=2', color='green')
plt.plot(x,sig3, label='a=3', color='blue')
plt.plot(x,sig4, label='a=4', color='orange')
plt.plot(x,sig5, label='a=5', color='cyan')
plt.xlabel('x')
plt.ylabel('$\sigma(x)$')
plt.legend()
plt.title('Sigmoid(ax)')
plt.grid(True)
plt.savefig("Adaptive Sigmoid")
plt.show()
plt.close()