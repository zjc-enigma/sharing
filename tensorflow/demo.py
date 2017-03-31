import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random


# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50


# make training data
def create_toy_data(func, low=0, high=1., n=100, std=1.):
    # sample points
    x = np.random.uniform(low, high, n)
    t = func(x) + np.random.normal(scale=std, size=n)
    #t = func(x)
    return x, t


def func(x):
    return np.sin(2 * np.pi * x)


train_X, train_Y = create_toy_data(func)
n_samples = train_X.shape[0]
test_X = np.linspace(0, 1, 100)



plt.scatter(train_X,
            train_Y,
            alpha=0.5,
            color="blue",
            label="observation")

plt.plot(test_X,
        func(test_X),
        color="red",
        label="sin$(2\pi x)$")

plt.show()


# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights : init to random values
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")


# Construct a linear model
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred - Y, 2))/(2*n_samples)
