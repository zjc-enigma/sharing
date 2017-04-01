import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
rng = np.random


# Parameters
learning_rate = 0.01
training_epochs = 5000
display_step = 50
batch_size = 50


# make training data
def create_toy_data(func, low=0, high=1., n=1000, std=1.):
    # sample points
    x = np.random.uniform(low, high, n)
    t = func(x) + np.random.normal(scale=std, size=n)
    return train_test_split(x, t, test_size=0.2, random_state=42)


def func(x):
    return 3.56 * x + 1.28

train_X, test_X, train_Y, test_Y = create_toy_data(func)
n_samples = train_X.shape[0]


plt.scatter(train_X,
            train_Y,
            alpha=0.5,
            color="blue",
            label="observation")


real_X = np.linspace(0, 1, 100)
plt.plot(real_X,
         func(real_X),
         color="black",
         label="3.56*x + 1.28")


def basis_func(X):
    return tf.sigmoid()
# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights : init to random values
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

w_h = tf.histogram_summary("weights", W)
b_h = tf.histogram_summary("bias", b)

# Construct a linear model
with tf.name_scope("model") as scope:
    pred = tf.add(tf.mul(X, W), b)


# Mean squared error
with tf.name_scope("cost_function") as scope:
    cost_func = tf.reduce_sum(tf.pow(pred - Y, 2))/(2*n_samples)
    tf.scalar_summary("cost_function", cost_func)

with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_func)


# init variables
init = tf.initialize_all_variables()

# merge all summaries into a single operator
merged_summary_op = tf.merge_all_summaries()

# lanch graph
with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.train.SummaryWriter("/tmp/tensorlog", graph=sess.graph)

    # training 
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples/batch_size)

        for i in range(total_batch):

            begin_idx = batch_size * i
            batch_xs = train_X[begin_idx:begin_idx+batch_size, ]
            batch_ys = train_Y[begin_idx:begin_idx+batch_size, ]
            _, batch_cost = sess.run([optimizer, cost_func], feed_dict={X:batch_xs,
                                                                        Y:batch_ys})

            avg_cost += batch_cost / total_batch
            summary_str = sess.run(merged_summary_op, feed_dict={X:batch_xs,
                                                                 Y:batch_ys})
            summary_writer.add_summary(summary_str, epoch*total_batch + i)


        # display log 
        if (epoch+1) % display_step == 0:
            print("Epoch:{}".format(epoch+1),
                  "cost={}".format(avg_cost),
                  "W={}".format(sess.run(W)),
                  "b={}".format(sess.run(b)))


    print("Optimization Finished!")
    training_cost = sess.run(cost_func, feed_dict={X:train_X, Y:train_Y})
    print("Training cost={}".format(training_cost),
          "W={}".format(sess.run(W)),
          "b={}".format(sess.run(b)))


    # using test set to validate
    testing_cost = sess.run(cost_func, feed_dict={X:test_X, Y:test_Y})
    print("Testing cost={}".format(testing_cost))
    print("Absolute mean square loss difference:{}"
          .format(abs(training_cost-testing_cost)))

    # plot
    plt.plot(train_X, sess.run(W)*train_X + sess.run(b), label="fitted curve")
    plt.plot(test_X, test_Y, 'ro', label="Testing data")
    plt.show()














