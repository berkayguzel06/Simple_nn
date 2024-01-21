from nodegrad.nn import MultiLayerNodes

n = MultiLayerNodes(3, [4,4,1])

print("------------------------------")
data = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
targets = [1.0, -1.0, -1.0, 1.0]
ypred = [n(x) for x in data]
for k in range(20):
  
  # forward pass
  ypred = [n(x) for x in data]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(targets, ypred))
  
  # backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()
  
  # update
  for p in n.parameters():
    p.data += -0.1 * p.grad
  
  print(k, loss.data)


print(ypred)