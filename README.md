# Road to Pytorch (Linear Regression)
Personal learning notes of Pytorch basics, contens were mostly from Daniel Bourke. Wish to learn more of the details, please check his own [Youtube channel](https://www.youtube.com/watch?v=Z_ikDlimN6A, 'YT Link').
## Contents
- ### [**Linear Regression**](#LR)
  - [**Import Pytorch and Data Preparation**](#Settings)
  - [**Visualization**](#Visual)
  - [**Linear Regression Model**](#LRModel)

<h2 id="LR">🏷 Linear Regression</h2>
<h3 id="Settings">Import Pytorch and Data Settings</h3>

First step of any Python coding is probably `import` libraries/packages.
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

<h3 id="Visual">Visualization</h3>

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

<h3 id="LRModel">Linear Regression Model</h3>
