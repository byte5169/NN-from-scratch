# Relative imports (as in from .. import mymodule) only work in a package.
# To import 'mymodule' that is in the parent directory of your current module with use the code below
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


### using Layer, Activation modules and create_data function we have created
from utils.data_generation import create_spin_data, visualize_data
from model.layer import Layer
from model.activation import ReLU, Softmax
from model.metrics import CatCrossEntropy, Accuracy

# generating data
X, y = create_spin_data(100, 3)

### Defining Model
# creating first layer(dense) with 2 inputs and 3 outputs
# create ReLU activation
# creating second layer(dense) with 3 inputs and 3 outputs
# create Softmax activation
# create loss function
layer1 = Layer(2, 3)
activation1 = ReLU()
layer2 = Layer(3, 3)
activation2 = Softmax()
loss_func = CatCrossEntropy()
acc = Accuracy()

# passing data through 1st layer
layer1.forward(X)
# forward pass 1st layer output through ReLU activation function
activation1.forward(layer1.output)
# forward pass of outputs of ReLu activation
layer2.forward(activation1.output)
# forward pass of 2nd layer outputs through Softmax function
activation2.forward(layer2.output)
# forward pass of ouput of 2nd layer through loss function, return loss
loss = loss_func.calc(activation2.output, y)
accs = acc.calc(activation2.output, y)

print("Results:\n", activation2.output[:3])
print("CrossEntLoss:\n", loss)
print("Accuracy:\n", accs)
