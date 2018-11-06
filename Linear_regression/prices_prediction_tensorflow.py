'''
Predicting prices with Boston housing dataset
'''
# importing libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading data and extractiong features and labels
boston = load_boston()
data = np.array(boston.data)
target = np.array(boston.target)

# Normalizing data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.25, random_state=100)
m = X_train.shape[0]
n_dim = X_train.shape[1]
print("number of training examples: ", m)
print("number of features: ", n_dim)

# function for creating placeholders
def create_placeholders(n_dim):
	X = tf.placeholder(tf.float32, [None, n_dim])
	y = tf.placeholder(tf.float32, [None,])
	return X, y

# function for initializing our parameters
def initialize_parameters(n_dim):
	w = tf.Variable(tf.zeros([n_dim, 1]))
	b = tf.Variable(tf.zeros([1,1]))
	return w, b

# Function for computing our hypothesis function
def model(X, w, b):
	predictions = tf.add(tf.matmul(X, w), b)
	return predictions

# Function for computing the cost
def compute_cost(predictions, y):
	loss = tf.reduce_mean(tf.square(predictions-y))
	return loss

# Setting hyperparameters
learning_rate = 0.001
n_epochs = 10000

# Creating the computational graph
with tf.name_scope ("LinearRegression") as scope:
	## Creating placeholders and initializing variables
	X, y = create_placeholders(n_dim)
	w, b = initialize_parameters(n_dim)
	## Defining the model
	predictions = model(X, w, b)
with tf.name_scope ("LossFunction") as scope:
	## loss function
	cost_op = compute_cost(predictions, y)
## Setting the optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost_op)
## Initializing the tensors
init = tf.global_variables_initializer()

#visualizing with tensoboard
## Create a summary to monitor the cost
loss_summary = tf.summary.scalar("loss",cost_op)
## Create a summary to monitor the parameters
w_ = tf.summary.histogram("W",w)
b_ = tf.summary.histogram("b",b)
## Merge all summaries into a single op
merged_op = tf.summary.merge_all()

# Running the computational graph
sess=tf.Session()
sess.run(init)
costs = []

#op to write logs to TensorBoard
writer_tensorboard = tf.summary.FileWriter("/home/ibtissem/stage_datavora/trials/LR_logs", tf.get_default_graph())

## Training the model and updating parameters
for epoch in range(n_epochs):
	_, cost_epoch = sess.run([optimizer, cost_op], feed_dict={X:X_train, y:y_train})
	costs.append(cost_epoch)
	print(cost_epoch)

# Plot of the costs
plt.plot(range(len(costs)), costs)
plt.axis([0, n_epochs, 0, max(costs)])
plt.show()

# Evaluating the model
pred_test = sess.run(predictions, feed_dict={X: X_test})
mse = compute_cost(pred_test, y_test)
print("MSE: %.4f"%sess.run(mse))

# plot of the line of best fit
fig, ax = plt.subplots()
ax.scatter(y_test, pred_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('predicted')
plt.show()

sess.close()


