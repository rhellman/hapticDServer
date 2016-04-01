# Test deepNets on matlab formated haptic data

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

import scipy.io as sio


# load formated data from .mat
mat_contents = sio.loadmat('sampleData_wNetChangeLabels.mat')
train_dataset  = np.array(mat_contents['train_data']).astype(np.float32)
train_labels = np.array(mat_contents['train_label']).astype(np.float32)
valid_dataset = np.array(mat_contents['test_data']).astype(np.float32)
valid_labels = np.array(mat_contents['test_label']).astype(np.float32)
print('Training set  - ', train_dataset.shape, train_labels.shape)
print('Test set      - ', valid_dataset.shape, ' ',valid_labels.shape)

sample_size = train_dataset.shape[1]
num_labels = train_labels.shape[1]

CL_1 = np.array([6, 7, 8 ,10])  #Imp(:, 7) + Imp(:, 8) + Imp(:, 9) + Imp(:,10);
CL_2 = np.array([0, 1, 2])      #Imp(:, 1) + Imp(:, 2) + Imp(:, 3);
CL_3 = np.array([10, 11, 12])   #Imp(:,11) + Imp(:,12) + Imp(:,13);
CL_4 = np.array([3, 4, 5])      #Imp(:, 4) + Imp(:, 5) + Imp(:, 6);
CL_5 = np.array([13, 14, 15])   #Imp(:,14) + Imp(:,15) + Imp(:,16);
CL_6 = np.array([16, 17, 18])   #Imp(:,17) + Imp(:,18) + Imp(:,19);

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


batch_size = 128
hidden_size = 120
cluster_output = hidden_size/6
hidden_1_size = 512
hidden_2_size = 512
beta = .1
SEED = None
dropoutPercent = 0.25

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, sample_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  #tf_test_dataset = tf.constant(test_dataset)
  
  # Variables
  weights_cl_1 = tf.Variable(tf.truncated_normal([CL_1.shape[0], cluster_output], stddev=0.1))
  biases_cl_1  = tf.Variable(tf.truncated_normal([cluster_output], stddev=0.1))
  weights_cl_2 = tf.Variable(tf.truncated_normal([CL_2.shape[0], cluster_output], stddev=0.1))
  biases_cl_2  = tf.Variable(tf.truncated_normal([cluster_output], stddev=0.1))
  weights_cl_3 = tf.Variable(tf.truncated_normal([CL_3.shape[0], cluster_output], stddev=0.1))
  biases_cl_3  = tf.Variable(tf.truncated_normal([cluster_output], stddev=0.1))
  weights_cl_4 = tf.Variable(tf.truncated_normal([CL_4.shape[0], cluster_output], stddev=0.1))
  biases_cl_4  = tf.Variable(tf.truncated_normal([cluster_output], stddev=0.1))
  weights_cl_5 = tf.Variable(tf.truncated_normal([CL_5.shape[0], cluster_output], stddev=0.1))
  biases_cl_5  = tf.Variable(tf.truncated_normal([cluster_output], stddev=0.1))
  weights_cl_6 = tf.Variable(tf.truncated_normal([CL_6.shape[0], cluster_output], stddev=0.1))
  biases_cl_6  = tf.Variable(tf.truncated_normal([cluster_output], stddev=0.1))
  weights_1    = tf.Variable(tf.truncated_normal([hidden_size, hidden_1_size], stddev=0.1))
  biases_1     = tf.Variable(tf.zeros([hidden_1_size]))
  #weights_2 = tf.Variable(tf.truncated_normal([hidden_1_size, hidden_2_size]))
  #biases_2  = tf.Variable(tf.zeros([hidden_2_size]))
  #weights_3 = tf.Variable(tf.truncated_normal([hidden_2_size, hidden_3_size]))
  #biases_3  = tf.Variable(tf.zeros([hidden_3_size]))
  # Output
  weights_o = tf.Variable(tf.truncated_normal([hidden_1_size, num_labels], stddev=0.1))
  biases_o  = tf.Variable(tf.zeros([num_labels]))

  def model(data, train=False):
    cl_1 = tf.nn.relu(tf.matmul(tf.transpose(tf.gather(tf.transpose(data),CL_1)), weights_cl_1) + biases_cl_1)
    cl_2 = tf.nn.relu(tf.matmul(tf.transpose(tf.gather(tf.transpose(data),CL_2)), weights_cl_2) + biases_cl_2)
    cl_3 = tf.nn.relu(tf.matmul(tf.transpose(tf.gather(tf.transpose(data),CL_3)), weights_cl_3) + biases_cl_3)
    cl_4 = tf.nn.relu(tf.matmul(tf.transpose(tf.gather(tf.transpose(data),CL_4)), weights_cl_4) + biases_cl_4)
    cl_5 = tf.nn.relu(tf.matmul(tf.transpose(tf.gather(tf.transpose(data),CL_5)), weights_cl_5) + biases_cl_5)
    cl_6 = tf.nn.relu(tf.matmul(tf.transpose(tf.gather(tf.transpose(data),CL_6)), weights_cl_6) + biases_cl_6)
    cl = tf.concat(1,[cl_1, cl_2, cl_3, cl_4, cl_5, cl_6])
    if train:
      cl = tf.nn.dropout(cl, dropoutPercent, seed= SEED)

    hidden = tf.nn.relu(tf.matmul(cl, weights_1) + biases_1)
    if train:
      hidden = tf.nn.dropout(hidden, dropoutPercent, seed= SEED)

    #hidden = tf.nn.relu(tf.matmul(hidden, weights_2) + biases_2)
    #if train:
    #  hidden = tf.nn.dropout(hidden, dropoutPercent, seed= SEED)
    #hidden = tf.nn.relu(tf.matmul(hidden, weights_3) + biases_3)
    #if train:
    #  hidden = tf.nn.dropout(hidden, dropoutPercent, seed= SEED)
    return tf.matmul(hidden, weights_o) + biases_o

  logits = model(tf_train_dataset, True)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  regularized_loss = loss + beta*( tf.nn.l2_loss(weights_cl_1) + tf.nn.l2_loss(weights_cl_2) +tf.nn.l2_loss(weights_cl_3) 
      +tf.nn.l2_loss(weights_cl_4) +tf.nn.l2_loss(weights_cl_5) +tf.nn.l2_loss(weights_cl_6) +tf.nn.l2_loss(weights_1) 
      + tf.nn.l2_loss(weights_o))
 #--------------------------------
   
  # Optimizer.
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.005, global_step, 250, 0.96)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(regularized_loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(model(tf_train_dataset))
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  #hidden_test_prediction = tf.nn.relu(tf.matmul(tf_test_dataset, weights_h) + biases_h)
  #test_prediction = tf.nn.softmax(tf.matmul(hidden_test_prediction, weights_o) + biases_o)
  

num_steps = 20001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f  Learning_rate: %f" % (step, l, learning_rate.eval()))
      print("Minibatch accuracy: %.1f%%"    % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%"   % accuracy(valid_prediction.eval(), valid_labels))
  #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))  


