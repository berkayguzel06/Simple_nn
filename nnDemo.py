from simplenn.model import Model
from simplenn.dense import Layer_Dense
from simplenn.engine import ReLU, Softmax, Accuracy_Categorical, Categorical_Cross_Entropy, Adam
from sklearn.datasets import make_blobs, make_circles , make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

X1, y1 = make_moons(n_samples=1000, noise=0.25)
X2, y2 = make_circles(n_samples=1000, noise=0.1)
X3, y3 = make_blobs(n_samples=1000)

noise_factor = 0.25
noise = np.random.normal(scale=noise_factor, size=X3.shape)
X3_noisy = X3 + noise

X_train, X_val, y_train, y_val = train_test_split(X3, y3, test_size=0.1)

plt.subplot(2,1,1)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, s=20, cmap='jet')
plt.subplot(2,1,2)
plt.scatter(X_val[:,0], X_val[:,1], c=y_val, s=20, cmap='jet')
plt.show()

data, target = X_train, y_train
validation_data, validation_target = X_val, y_val
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
model.train(data, target, epochs=1000, print_every=100, validation=(validation_data, validation_target))
