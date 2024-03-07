import numpy as np
import cvxpy as cp  # Library for convex optimization
import matplotlib.pyplot as plt
from scipy.fftpack import dct  # Discrete Cosine Transform
from scipy.fftpack import idct  # Inverse Discrete Cosine Transform


# Step 1: Define your signal and its sparse representation in some transform domain
fig, ax = plt.subplots(2,1,figsize=(10,5))

# Example: Sparse signal in the DCT domain
n = 4096
t = np.linspace(0, 1, n)
signal = np.cos(2 * 97 * np.pi * t) + np.cos(2 * 110 * np.pi * t) + np.cos(2 * 5 * np.pi * t) + + np.cos(2 * 200 * np.pi * t)
ax[0].plot(signal, color='black')
sparse_coeffs = dct(signal, norm='ortho')  # Sparse coefficients

# Step 2: Define or generate the measurement matrix
m = 50  # Number of measurements
n = len(signal)  # Dimensionality of the signal
print(n)
A = np.random.randn(m, n)  # Random Gaussian measurement matrix

# Step 3: Acquire measurements
y = np.dot(A, sparse_coeffs)  # Compressed measurements

# Step 4: Sparse signal recovery using Basis Pursuit (BP)
x_hat = cp.Variable(n)  # Sparse signal variable
objective = cp.Minimize(cp.norm(x_hat, 1))
constraints = [A @ x_hat == y]
problem = cp.Problem(objective, constraints)
problem.solve()

recovered_coeffs = np.array(x_hat.value)

# Step 5: Inverse transform to reconstruct the original signal
reconstructed_signal = idct(recovered_coeffs, norm='ortho')

# # Display or save reconstructed signal
ax[1].plot(reconstructed_signal, color = 'r')
plt.show()