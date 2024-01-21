import random
from nodegrad.node import Node

class Neuron:
    def __init__(self, inputs):
        self.w = [Node(random.uniform(-1,1)) for _ in range(inputs)]
        self.b = Node(random.uniform(-1,1))
    def __call__(self, x):
        out = sum(wi*xi for wi,xi in zip(self.w, x)) + self.b
        out = out.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, inputs, outputs):
        self.neurons = [Neuron(inputs) for _ in range(outputs)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        params = []
        for neuron in self.neurons:
            param = neuron.parameters()
            params.extend(param)
        return params
    
class MultiLayerNodes:
    def __init__(self, inputs, outputs):
        size = [inputs] + outputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(outputs))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        params = []
        for neuron in self.layers:
            param = neuron.parameters()
            params.extend(param)
        return params