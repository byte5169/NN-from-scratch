from neural_net.utils.data_generation import create_spin_data, visualize_data
from neural_net.model.layer import Layer
from neural_net.model.activation import ReLU
from neural_net.model.metrics import Accuracy, Softmax_CatCrossEntropy
from neural_net.model.optimizer import SGD, Adagrad, Adam, RMSprop

# generating data
X, y = create_spin_data(100, 3)

### Defining Model
# creating first layer(dense) with 2 inputs and 3 outputs
# create ReLU activation
# creating second layer(dense) with 3 inputs and 3 outputs
# create Softmax combined with loss and activation
# calculate accuracy
# create optimizer
layer1 = Layer(2, 64)
activation1 = ReLU()
layer2 = Layer(64, 3)
loss_activation = Softmax_CatCrossEntropy()
acc = Accuracy()
optim = Adam(lr=0.005, decay=5e-7)

# training loop
for epoch in range(10001):
    # passing data through 1st layer
    layer1.forward(X)
    # forward pass 1st layer output through ReLU activation function
    activation1.forward(layer1.output)
    # forward pass of outputs of ReLu activation
    layer2.forward(activation1.output)
    # forward pass of ouput of 2nd layer through activtion/loss function, return loss
    loss = loss_activation.forward(layer2.output, y)
    accs = acc.calc(loss_activation.output, y)

    if not epoch % 100:
        print(
            f"epoch: {epoch}, loss: {loss:.3f}, acc: {accs:.3f}, lr: {optim.current_lr:.5f}"
        )

    # backward pass
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # update w, b
    optim.pre_update_params()
    optim.update_params(layer1)
    optim.update_params(layer2)
    optim.post_update_params()

# # gradients
# print(layer1.dweights)
# print(layer1.dbiases)
# print(layer2.dweights)
# print(layer2.dbiases)