# Road to Pytorch (Linear Regression)
Personal learning notes of Pytorch basics, contens were mostly from Daniel Bourke. Wish to learn more of the details, please check his own [Youtube channel](https://www.youtube.com/watch?v=Z_ikDlimN6A, 'YT Link'). I want to put everything together for myself and it will be nice if the contents can help those who want to learn about Pytorch, so GOOD LUCK~~
## Contents
- ### [**Import Pytorch and Data Preparation**](#Settings)
- ### [**Visualization**](#Visual)
- ### [**Linear Regression Model**](#LRModel)

<h2 id="Settings">Import Pytorch and Data Settings</h2>

First step of any Python coding is probably **import** libraries/packages.
```python
import torch
from torch import nn
import matplotlib.pyplot as plt
```
> Import `torch` for accessing the modules under Pytorch, `nn` for later model class inheritance.

There is no better way of using [linear regression](https://en.wikipedia.org/wiki/Linear_regression, 'Wiki of LR') (LR) for training of LR model. 

LR is generally in a form of `y = weight * X + bias`, which we shall create the function first.
```python
# set fixed parameters
weight = 0.7
bias = 0.3

# create X and y
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim = 1)
y = weight * X + bias
```
Afterwards, we shall split the data into training and testing dataset, which will be in a ratio of 8:2.
```python
# split data into training set and testing set
train_split = int(len(X) * 0.8)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
```
> The length of `X_train` (and `y_train`) will be **(1 / 0.02) * (0.8) = 40**, `X_test` (`y_test`) will be **(1 / 0.02) * (0.2) = 10**.

<h2 id="Visual">Visualization</h2>

Visualize the data points for clarity. The plotting function will plot the training data points, testing data points and the predicted data points if they existed.
```python
# visualize
def plot_prediction(train_data = X_train,
                    train_label = y_train,
                    test_data = X_test,
                    test_label = y_test,
                    predictions = None):
    plt.figure(figsize = (10, 7))
    plt.scatter(train_data, train_label, c = 'b', s = 4, label = 'Training Data')
    plt.scatter(test_data, test_label, c = 'g', s = 4, label = 'Testing Data')
    if predictions is not None:
        plt.scatter(test_data, predictions, c = 'r', s = 4, label = 'Predictions')
    plt.legend(prop = {'size':14})
```
> The training set will be plotted as blue, testing set will be green and the predictions will be red.

<h2 id="LRModel">Linear Regression Model</h2>

Before building the model, here are some Pytorch model essentials：
* `torch.nn` - contains all the buildings for computational graphs (aka nn)
* `torch.nn.parameter` - what parameters should our model try and learn, often a pytorch layer will set these for us
* `torch.nn.Module` - the base class for all neural network modules, if you subclass it, you should overwrite `forward()`
* `torch.optim` - this is where the optimizer in pytorch live, they will help with gradient descent
* `def forward()` - defines what happenens in the forward computaions

First of all, we shall build the model **Class** called `LinearRegressionModel` that inherit from `nn.Module`.
```python
# create linear regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in pytorch inherhit from nn.Module
  ...
```
Afterwards, initialize the parameters that we gonna use, which are bias and weight.
```python
def __init__(self):
    super().__init__()

    # initialize the parameters (parameters that will be used into the computations)
    self.weights = nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))
    self.bias = nn.Parameter(torch.randn(1,
                                        requires_grad=True,
                                        dtype=torch.float))
```
> Both bias and weight will be randomly generated.

In the end, overwrite the `forward()`, which will return the calculation results of linear regression function.
```python
# forward() defines the computaion in the model
def forward(self, x:torch.Tensor) -> torch.Tensor: # <- "x" is the input data
    return self.weights * x + self.bias # linear regression model
```
> Take **x** as input, the return value type should be `torch.Tensor` as well.
