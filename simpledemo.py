from simplenn.model import Model
from simplenn.dense import Layer_Dense, Layer_Dropout
from simplenn.activation import ReLU, Softmax
from simplenn.accuracy import Accuracy_Categorical, Accuracy_Regression
from simplenn.loss import Categorical_Cross_Entropy, Mean_Square_Error
from simplenn.optimizer import Adam, SGD
from sklearn.datasets import make_blobs, make_circles , make_moons
from nnfs.datasets import sine_data, spiral_data, vertical_data
import matplotlib.pyplot as plt
import nnfs

nnfs.init()
X_val, y_val = sine_data(samples=100)
X1_val, y1_val = vertical_data(samples=100,classes=3)
X2_val, y2_val = spiral_data(samples=100,classes=3)
X3_val, y3_val = make_moons(n_samples=100, noise=0.1)
X4_val, y4_val = make_circles(n_samples=100, noise=0.1)
X5_val, y5_val = make_blobs(n_samples=100)

X, y = sine_data(samples=1000)
X1, y1 = vertical_data(samples=1000,classes=3)
X2, y2 = spiral_data(samples=1000,classes=3)
X3, y3 = make_moons(n_samples=1000, noise=0.25)
X4, y4 = make_circles(n_samples=1000, noise=0.1)
X5, y5 = make_blobs(n_samples=1000)
plt.subplot(2,3,1)
plt.scatter(X1[:,0], X1[:,1], c=y1, s=20, cmap='jet')
plt.subplot(2,3,2)
plt.scatter(X2[:,0], X2[:,1], c=y2, s=20, cmap='jet')
plt.subplot(2,3,3)
plt.scatter(X3[:,0], X3[:,1], c=y3, s=20, cmap='jet')
plt.subplot(2,3,4)
plt.scatter(X4[:,0], X4[:,1], c=y4, s=20, cmap='jet')
plt.subplot(2,3,5)
plt.scatter(X5[:,0], X5[:,1], c=y5, s=20, cmap='jet')
plt.subplot(2,3,6)
plt.scatter(X, y, s=20, cmap='jet')
plt.show()


data = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
targets = [1.0, -1.0, -1.0, 1.0]

model = Model()
model.add(Layer_Dense(2,16))
model.add(ReLU())
model.add(Layer_Dense(16,3))
model.add(Softmax())

model.set(
    loss = Categorical_Cross_Entropy(),
    optimizer = Adam(lr=0.05),
    accuracy = Accuracy_Categorical()
)
model.finalize()
model.train(X5, y5, epochs=1000, print_every=100,validation=(X5_val, y5_val))


"""
model2 = Model()
model.add(Layer_Dense(1,64))
model.add(ReLU())
model.add(Layer_Dense(64,64))
model.add(ReLU())
model.add(Layer_Dense(64,1))
model2.set(
    loss = Mean_Square_Error(),
    optimizer = Adam(lr=0.005,decay=1e-3),
    accuracy = Accuracy_Regression()
)
model2.finalize()
model2.train(X, y, epochs=1000, print_every=100, validation=(X_val, y_val))
"""
