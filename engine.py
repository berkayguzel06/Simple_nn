import math
import random
from typing import Any

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

class Node:
    def __init__(self, data, _connect=(), _op='', label =''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_connect)
        self._backward = lambda:None
        self._op = _op
        self.label = label
    def __repr__(self):
        return f"Node ( Label = {self.label}, Data = {self.data}, Grad = {self.grad} )\n"
    def __add__(self, node):
        node = node if isinstance(node, Node) else Node(node)
        out = Node(self.data+node.data, (self, node), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            node.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __radd__(self, node):
        return self + node
    def __pow__(self,node):
        assert isinstance(node, (int, float)), 'only int/float'
        out = Node(self.data**node, (self,), f'**{node}')
        def _backward():
            self.grad += node * (self.data ** (node-1)) * out.grad
        out._backward=_backward
        return out
    def __mul__(self, node):
        node = node if isinstance(node, Node) else Node(node)
        out = Node(self.data*node.data, (self, node), '*')
        def _backward():
            self.grad += node.data * out.grad
            node.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __rmul__(self,node):
        return self*node
    def __truediv__(self,node):
        return self * node**-1
    def __neg__(self):
        return self * -1
    def __sub__(self, node):
        return self + (-node)
    def exp(self):
        x = self.data
        out = Node(math.exp(x),(self,),'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward=_backward
        return out
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Node(t, (self, ))
        def _backward():
            self.grad += (1-t**2)*out.grad

        out._backward = _backward
        return out
    def backward(self):     
        connections = []
        connected = set()
        def build_connections(node):
            if node not in connected:
                connected.add(node)
                for connection in node._prev:
                    build_connections(connection)
                connections.append(node)
        build_connections(self)

        self.grad = 1.0
        for node in reversed(connections):
            node._backward()

    def connections(self):
        connections = []
        connected = set()
        def build_connections(node):
            if node not in connected:
                connected.add(node)
                for connection in node._prev:
                    build_connections(connection)
                connections.append(node)
        build_connections(self)
        for l in connections:
            print(l)


# inputs x1,x2
x1 = Node(2.0, label='x1')
x2 = Node(0.0, label='x2')
# weights w1,w2
w1 = Node(-3.0, label='w1')
w2 = Node(1.0, label='w2')
# bias of the neuron
b = Node(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'
o.connections()
o.backward()
print("-----------------------")
o.connections()
print("-----------NEURON-----------")
x = [2.0, 3.0, -1.0]
n = MultiLayerNodes(3, [4,4,1])
print(n(x))

print("------------------------------")
data = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
targets = [1.0, -1.0, -1.0, 1.0]
print("PREDS")
y_pred = [n(x) for x in data]
print(y_pred)

print("LOSS")
loss = sum([(out - target)**2 for target, out in zip(targets, y_pred)])
print(loss)
loss.backward()
print(n.layers[0].neurons[0].w[0])

print("PARAMETERS")
print(f"Total {len(n.parameters())} parameters.")
print(n.parameters())