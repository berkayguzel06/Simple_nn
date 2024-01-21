import math


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


