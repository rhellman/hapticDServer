# Test deepNets on matlab formated haptic data

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

import scipy.io as sio
import sys, os, os.path, inspect

# Need to ensure that the training data will be used from deepNetTest.py 
# package location
filedir = inspect.getframeinfo(inspect.currentframe())[0]
filedir = os.path.dirname(os.path.abspath(filedir))

if __name__ == "__main__":
  mat_contents = sio.loadmat( filedir + '/sampleData_wNetChangeLabels.mat')
  train_dataset  = np.array(mat_contents['train_data']).astype(np.float32)
  train_labels = np.array(mat_contents['train_label']).astype(np.float32)
  valid_dataset = np.array(mat_contents['test_data']).astype(np.float32)
  valid_labels = np.array(mat_contents['test_label']).astype(np.float32)
  print('Training set  - ', train_dataset.shape, train_labels.shape)
  print('Test set      - ', valid_dataset.shape, ' ',valid_labels.shape)

  sample_size = train_dataset.shape[1]
  num_labels  = train_labels.shape[1]

else:
  sample_size = 24
  num_labels  = 2
  valid_dataset = np.zeros([10,sample_size]).astype(np.float32)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


batch_size = 128
hidden_size = 1024
hidden_1_size = hidden_size
hidden_2_size = hidden_size
hidden_3_size = hidden_size
beta = .01
SEED = None
dropoutPercent = 0.5

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, sample_size))
  tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  
  tf_predict_label = tf.placeholder(tf.float32, shape=(1,sample_size))
  
  # Variables
  weights_1 = tf.Variable(tf.truncated_normal([sample_size, hidden_1_size], stddev=0.1), name="weights_1")
  biases_1  = tf.Variable(tf.zeros([hidden_1_size]), name="biases_1")
  weights_2 = tf.Variable(tf.truncated_normal([hidden_1_size, hidden_2_size]), name="weights_2")
  biases_2  = tf.Variable(tf.zeros([hidden_2_size]), name="biases_2")
  weights_3 = tf.Variable(tf.truncated_normal([hidden_2_size, hidden_3_size]), name="weights_3")
  biases_3  = tf.Variable(tf.zeros([hidden_3_size]), name="biases_3")
  # Output
  weights_o = tf.Variable(tf.truncated_normal([hidden_3_size, num_labels], stddev=0.1), name="weights_o")
  biases_o  = tf.Variable(tf.zeros([num_labels]), name="biases_o")



  def model(data, train=False):
    hidden = tf.nn.relu(tf.matmul(data, weights_1) + biases_1)
    if train:
      hidden = tf.nn.dropout(hidden, dropoutPercent, seed= SEED)
    hidden = tf.nn.relu(tf.matmul(hidden, weights_2) + biases_2)
    if train:
      hidden = tf.nn.dropout(hidden, dropoutPercent, seed= SEED)
    hidden = tf.nn.relu(tf.matmul(hidden, weights_3) + biases_3)
    if train:
      hidden = tf.nn.dropout(hidden, dropoutPercent, seed= SEED)
    return tf.matmul(hidden, weights_o) + biases_o


  logits = model(tf_train_dataset, True)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  regularized_loss = loss + beta*( tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
       + tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_o))
  #--------------------------------
   
  # Optimizer.
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.001, global_step, 250, 0.96)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(regularized_loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(model(tf_train_dataset))
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  data_prediction  = tf.nn.softmax(model(tf_predict_label))
  #hidden_test_prediction = tf.nn.relu(tf.matmul(tf_test_dataset, weights_h) + biases_h)
  #test_prediction = tf.nn.softmax(tf.matmul(hidden_test_prediction, weights_o) + biases_o)

  
predictHasLoaded = 0;
session = tf.Session(graph=graph)
def predict():
  global predictHasLoaded
  global session
  if predictHasLoaded == 0:
      saver = tf.train.Saver()
      saver.restore(session, filedir + "/clusterNN.ckpt")  
  predictHasLoaded += 1
  input_data = np.array(range(24))
  input_data = np.reshape(input_data, (1,-1))
  #print("input_data.shape:", input_data.shape)
  feed_dict = {tf_predict_label : input_data}
  predictions = session.run([data_prediction], feed_dict=feed_dict)
    

# to be used for training 
num_steps = 5001
def main(train = 0):
  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
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
      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 500 == 0):
        print("Minibatch loss at step %d: %f  Learning_rate: %f" % (step, l, learning_rate.eval()))
        print("Minibatch accuracy: %.1f%%"    % accuracy(predictions, batch_labels))
        print("Validation accuracy: %.1f%%"   % accuracy(valid_prediction.eval(), valid_labels))
    #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))  
    save_path = saver.save(session, os.getcwd() + '/clusterNN.ckpt')

#def loadmodel(filename):
#  pass
  

if __name__ == "__main__":
  main()




