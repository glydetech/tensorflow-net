# tensorflow-net
A wrapper around a tensorflow model to allow modifying layers and weights

Create and print a new network with 3 inputs, 2 outputs and 2 hidden layers of size 5 and 4:

```python
from tensorflow_net.net import Net
mynetwork = Net([3, 5, 4, 2])
mynetwork.print()
```

You'll see a dump of the layer sizes, types and weights:

```
sizes: [3, 5, 4, 2]
layer: 0 Tensor("InputData/X:0", shape=(?, 3), dtype=float32)
[]
layer: 1 Tensor("FullyConnected/Tanh:0", shape=(?, 5), dtype=float32)
[[ 0.02885317  0.01698384 -0.00406004  0.01117233  0.00928431]
 [ 0.01596719  0.01427222  0.03723403  0.02471406 -0.00254035]
 [ 0.01412388 -0.01161906 -0.00968075 -0.03335618  0.02104566]]
layer: 2 Tensor("FullyConnected_1/Tanh:0", shape=(?, 4), dtype=float32)
[[-0.01477276  0.00042995 -0.02129694  0.00329449]
 [-0.00831893 -0.01271056 -0.00871137 -0.01092652]
 [ 0.03850821 -0.02936289 -0.00866483  0.00699756]
 [-0.01035893  0.02504826  0.03606634  0.0153722 ]
 [ 0.03675643  0.00712568 -0.00766616  0.0287251 ]]
layer: 3 Tensor("FullyConnected_2/Softmax:0", shape=(?, 2), dtype=float32)
[[-0.01681465  0.0008236 ]
 [-0.00099722 -0.00149076]
 [ 0.0015666  -0.00728575]
 [-0.01928817  0.00525636]]
```

In code you'll find methods for the following:

* modifying individual weights
* adjusting weights by layer
* duplicating a network
* appending nodes to layers
* appending a new layer

