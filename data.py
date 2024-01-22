from nnfs.datasets import spiral_data, sine_data, vertical_data
import matplotlib.pyplot as plt

X,y = spiral_data(n_samples=100)

plt.scatter(X,y)
plt.show()