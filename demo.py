from data_prep import load_data_wrapper
from miniflow import *
import random
import matplotlib.pyplot as plt
import numpy as np

# load mnist data
training_data, validation_data, test_data = load_data_wrapper()

# build 3-layer network 784 x 30 x 10
n_features = 784
n_hidden = 30
n_output = 10

# input layer
X = Input()
# hidden layer
W1, b1 = Input(), Input()
l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
# output layer
W2, b2 = Input(), Input()
l2 = Linear(s1, W2, b2)
s2 = Sigmoid(l2)
# cost
Y = Input()
cost = MSE(Y, s2)

X_ = np.zeros(n_features)
Y_ = np.zeros(n_output)
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, n_output)
b2_ = np.zeros(n_output)

feed_dict = {
    X: X_,
    Y: Y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}
graph = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]


# hyper-parameters
epochs = 1000
batch_size = 5000
learning_rate = 0.18

# store train loss and validate loss
losses = {'train': [], 'validation': []}

# train the network
i = 0
while i < epochs:

    random.shuffle(training_data)
    batches = [
        training_data[k:k+batch_size]
        for k in range(0, len(training_data), batch_size)]

    for batch in batches:
        if i >= epochs: break
        i += 1

        # train data and compute train loss
        loss = 0
        for data in batch:
            X.value = data[0]
            Y.value = data[1]
            forward_and_backward(graph)
            sgd_update(trainables, learning_rate)
            loss += graph[-1].value
        losses['train'].append(loss/len(batch))

        # compute validate loss
        loss = 0
        for x, y in validation_data:
            X.value = x
            Y.value = y
            forward(graph)
            loss += graph[-1].value
        losses['validation'].append(loss/len(validation_data))

        print("Epoch: {}, Train Loss: {:.3f}, Validation Loss: {:.3f}".format(i,
                                                                              losses['train'][-1],
                                                                              losses['validation'][-1]))
# show graph of train loss and validation loss
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.show()

# run against test data
count = 0
for x, y in test_data:
    X.value = np.array(x).reshape(1, n_features)
    forward(graph)
    if np.argmax(s2.value) == y:
        count += 1
print('Recognized: {0} / {1}'.format(count, len(test_data)))
