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
classes = 3
X_val, y_val = sine_data(samples=100)
X1_val, y1_val = vertical_data(samples=100,classes=classes)
X2_val, y2_val = spiral_data(samples=100,classes=classes)
X3_val, y3_val = make_moons(n_samples=100, noise=0.2)
X4_val, y4_val = make_circles(n_samples=100, noise=0.1)
X5_val, y5_val = make_blobs(n_samples=100)

X, y = sine_data(samples=1000)
X1, y1 = vertical_data(samples=1000,classes=classes)
X2, y2 = spiral_data(samples=1000,classes=classes)
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

data, target = X5, y5
validation_data, validation_target = X5_val, y5_val
input_size = data.shape[1]
print(f"Input shape: {data.shape}, Target shape: {target.shape}")
print(f"Data size: {data.size}, Target size: {target.size}")
print(f"Input size: {input_size}, Output size: {target.shape}")

model = Model()
model.add(Layer_Dense(input_size,16))
model.add(ReLU())
model.add(Layer_Dense(16,3))
model.add(Softmax())

model.set(
    loss = Categorical_Cross_Entropy(),
    optimizer = Adam(lr=0.005, decay=1e-7),
    accuracy = Accuracy_Categorical()
)
model.finalize()
print(model.get_parameters())
model.train(data, target, epochs=1000, print_every=100, batch_size=100, validation=(validation_data, validation_target))
print(model.get_parameters())
