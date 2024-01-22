from simplenn.model import Model
from simplenn.dense import Layer_Dense, Layer_Dropout
from simplenn.activation import ReLU, Softmax
from simplenn.accuracy import Accuracy_Categorical
from simplenn.loss import Categorical_Cross_Entropy
from simplenn.optimizer import Adam
from nnfs.datasets import sine_data, spiral_data, vertical_data
import nnfs

nnfs.init()

X, y = spiral_data(samples=1000,classes=3)
X_val, y_val = spiral_data(samples=100,classes=3)
X, y = sine_data(samples=1000)
X_val, y_val = sine_data(samples=100)
X, y = vertical_data(samples=1000,classes=3)
X_val, y_val = vertical_data(samples=100,classes=3)

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
    optimizer = Adam(lr=0.05,decay=5e-7),
    accuracy = Accuracy_Categorical()
)
model.finalize()
model.train(X, y, epochs=100, print_every=1, validation=(X_val, y_val))