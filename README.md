# MiniFlow
This is a simple and extendable neutal network framework build through Udacity Deep learning Nanodegree.

# Usage
### Build network
```python
# define nodes
X, W, b = Input(), Input(), Input()
y = Input()
f = Linear(X, W, b)
a = Sigmoid(f)
cost = MSE(y, a)

# collect trainable nodes
trainables = [W, b]

# prepare value for nodes
X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2.], [3.]])
b_ = np.array([-3.])
y_ = np.array([1, 2])

# feed the value and get a topological sort order
feed_dict = {
    X: X_,
    y: y_,
    W: W_,
    b: b_,
}
graph = topological_sort(feed_dict)
```
### Train network
```python
# execute one forward and bakward pass
forward_and_backward(graph)

# update value of trainable nodes
sgd_update(trainables, learning_rate)
```
# Example
In `demo.py`, I write a simple example to recognize handwritten digits from [mnist](http://yann.lecun.com/exdb/mnist/), see more
usage example [here](https://github.com/CtheSky/MiniFlow/blob/master/demo.py).
