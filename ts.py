# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import math as m
import numpy as np
from PIL import Image

from Dataset import *

d = Dataset(784, 10)

d.add(d.imageToArray("minimnist/0.png", "L"), d.selectedOutputToArray(0))
d.add(d.imageToArray("minimnist/1.png", "L"), d.selectedOutputToArray(1))
d.add(d.imageToArray("minimnist/2.png", "L"), d.selectedOutputToArray(2))
d.add(d.imageToArray("minimnist/3.png", "L"), d.selectedOutputToArray(3))
d.add(d.imageToArray("minimnist/4.png", "L"), d.selectedOutputToArray(4))
d.add(d.imageToArray("minimnist/5.png", "L"), d.selectedOutputToArray(5))
d.add(d.imageToArray("minimnist/6.png", "L"), d.selectedOutputToArray(6))
d.add(d.imageToArray("minimnist/7.png", "L"), d.selectedOutputToArray(7))
d.add(d.imageToArray("minimnist/8.png", "L"), d.selectedOutputToArray(8))
d.add(d.imageToArray("minimnist/9.png", "L"), d.selectedOutputToArray(9))

d.printInput()
d.printOutput()

# N       = 64    # batch size
D_in    = 784
H       = 10
D_out   = 10

x = Variable(torch.Tensor(d.getInput().tolist()))
# print x
y = Variable(torch.Tensor(d.getOutput().tolist()), requires_grad=False)

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
# x = Variable(torch.randn(N, D_in))
# # print x
# y = Variable(torch.randn(N, D_out), requires_grad=False)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    # torch.nn.Linear(H, H),
    # torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    )

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 0.00001
for t in range(20000):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Variables containing the predicted and true
    # values of y, and the loss function returns a Variable containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Variable, so
    # we can access its data and gradients like we did before.
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

print model