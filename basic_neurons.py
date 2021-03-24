import numpy as np

# by using weights and biases - calculate 3 neurons
# generate random i, w, b
inputs = np.random.rand(3)
weights = [np.random.rand(3) for i in range(3)]
biases = np.random.rand(3)


### y = m*x + b, performing calculation
output1 = [
    inputs[0] * weights[0][0]
    + inputs[1] * weights[0][1]
    + inputs[2] * weights[0][2]
    + biases[0],
    inputs[0] * weights[1][0]
    + inputs[1] * weights[1][1]
    + inputs[2] * weights[1][2]
    + biases[1],
    inputs[0] * weights[2][0]
    + inputs[1] * weights[2][1]
    + inputs[2] * weights[2][2]
    + biases[2],
]
print("Results of 1st output: \n", output1)


### the same as above, but using for loops
output2 = []
# creating zipped list of weights and biases for each neuron
for n_w, n_b in zip(weights, biases):
    n_output = 0
    # input and weight of neuron
    for i, w in zip(inputs, n_w):
        # y = m*x + b
        n_output += i * w
    n_output += n_b
    output2.append(n_output)
print("Results of 2nd output: \n", output2)


### using dot product and adding hidden layers

i = np.random.randn(3, 4)
w1 = np.random.randn(3, 4)
w2 = np.random.randn(3, 3)
b = np.random.rand(3)
b2 = np.random.rand(3)

l1_layer = np.dot(i, w1.T) + b
l2_layer = np.dot(l1_layer, w2.T) + b2

print("Shapes of...")
print("Input layer:", i.shape)
print("1st weights:", w1.shape)
print("2nd weights:", w2.shape)
print("1st hidden layer:", l1_layer.shape)
print("1st hidden layer:", l1_layer.shape)
print("Results of 1st hidden layer\n", l1_layer)
print("Results of 2st layer\n", l2_layer)
