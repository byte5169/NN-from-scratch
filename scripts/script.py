# Relative imports (as in from .. import mymodule) only work in a package.
# To import 'mymodule' that is in the parent directory of your current module with use the code below
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


### using Layer, Activation modules and create_data function we have created
from utils.data_generation import create_data, visualize_data
from model.layer import Layer
from model.activation import ReLU, Softmax

# generating data
X, y = create_data(100, 3)

# creating first layer(dense) with 2 inputs and 3 outputs
layer1 = Layer(2, 3)
# create ReLU activation
activation1 = ReLU()
# creating second layer(dense) with 3 inputs and 3 outputs
layer2 = Layer(3, 3)
# create Softmax activation
activation2 = Softmax()
# passing data through 1st layer
layer1.forward(X)
# forward pass 1st layer output through ReLU activation function
activation1.forward(layer1.output)
# forward pass of outputs of ReLu activation
layer2.forward(activation1.output)
# forward pass of 2nd layer outputs through Softmax function
activation2.forward(layer2.output)


print("Results:\n", activation2.output[:3])