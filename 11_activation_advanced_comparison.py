# 11_activation_advanced_comparison.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

x = torch.linspace(-10, 10, 1000)

# Define activation functions
# ReLU
relu = np.maximum(0, x)
# Swish
swish = x * torch.sigmoid(x)
# GELU using approximate formula
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# Plot ReLU
plt.figure()
plt.plot(x, relu, label='ReLU')
plt.title("ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.grid()
plt.show()

# Plot Swish
plt.figure()
plt.plot(x, swish, label='Swish', color='orange')
plt.title("Swish Activation Function")
plt.xlabel("x")
plt.ylabel("Swish(x)")
plt.grid()
plt.show()

# Plot GELU
plt.figure()
plt.plot(x, gelu, label='GELU', color='green')
plt.title("GELU Activation Function")
plt.xlabel("x")
plt.ylabel("GELU(x)")
plt.grid()
plt.show()

# ALL Plot
x = torch.linspace(-5, 5, 1000)
relu = F.relu(x)
swish = x / (1 + np.exp(-x))
gelu = F.gelu(x)
plt.figure()
plt.plot(x.numpy(), relu.numpy(), label='ReLU')
plt.plot(x.numpy(), swish.numpy(), label='Swish')
plt.plot(x.numpy(), gelu.numpy(), label='GELU')
plt.legend()
plt.title("Activation Function Comparison")
plt.grid()
plt.show()