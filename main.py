#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import random
import numpy as np
import tensorflow as tf


NUM_CLASSES = 8
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', './data/train/data.txt', 'File name of train data')
flags.DEFINE_string('test', './data/test/data.txt', 'File name of train data')
flags.DEFINE_string('train_dir', './data', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 30, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

def inference(images_placeholder, keep_prob):
  def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

  with tf.name_scope('conv1') as scope:
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  with tf.name_scope('pool1') as scope:
    h_pool1 = max_pool_2x2(h_conv1)

  with tf.name_scope('conv2') as scope:
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  with tf.name_scope('pool2') as scope:
    h_pool2 = max_pool_2x2(h_conv2)

  with tf.name_scope('fc1') as scope:
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, rate = 1 - keep_prob)

  with tf.name_scope('fc2') as scope:
    W_fc2 = weight_variable([1024, NUM_CLASSES])
    b_fc2 = bias_variable([NUM_CLASSES])

  with tf.name_scope('softmax') as scope:
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  return y_conv

def loss(logits, labels):
  cross_entropy = -tf.reduce_sum(labels*tf.math.log(logits))
  tf.compat.v1.summary.scalar("cross_entropy", cross_entropy)
  return cross_entropy

def training(loss, learning_rate):
  train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
  return train_step

def accuracy(logits, labels):
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  tf.compat.v1.summary.scalar("accuracy", accuracy)
  return accuracy

if __name__ == '__main__':
  f = open(FLAGS.train, 'r')
  train_image = []
  train_label = []
  for line in f:
    line = line.rstrip()
    l = line.split()
    img = cv2.imread(l[0])
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    train_image.append(img.flatten().astype(np.float32)/255.0)
    tmp = np.zeros(NUM_CLASSES)
    tmp[int(l[1])] = 1
    train_label.append(tmp)
  train_image = np.asarray(train_image)
  train_label = np.asarray(train_label)
  f.close()

  f = open(FLAGS.test, 'r')
  test_image = []
  test_label = []
  for line in f:
    line = line.rstrip()
    l = line.split()
    img = cv2.imread(l[0])
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    test_image.append(img.flatten().astype(np.float32)/255.0)
    tmp = np.zeros(NUM_CLASSES)
    tmp[int(l[1])] = 1
    test_label.append(tmp)
  test_image = np.asarray(test_image)
  test_label = np.asarray(test_label)
  f.close()

  with tf.Graph().as_default():
    images_placeholder = tf.compat.v1.placeholder("float", shape=(None, IMAGE_PIXELS))
    labels_placeholder = tf.compat.v1.placeholder("float", shape=(None, NUM_CLASSES))
    keep_prob = tf.compat.v1.placeholder("float")

    logits = inference(images_placeholder, keep_prob)
    loss_value = loss(logits, labels_placeholder)
    train_op = training(loss_value, FLAGS.learning_rate)
    acc = accuracy(logits, labels_placeholder)


    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    sess.run(tf.initialize_all_variables())
    summary_op = tf.compat.v1.summary.merge_all()
    summary_writer = tf.compat.v1.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in range(FLAGS.max_steps):
      for i in range(int(len(train_image)/FLAGS.batch_size)):
        batch = FLAGS.batch_size*i
        sess.run(train_op, feed_dict={
          images_placeholder: train_image[batch:batch+FLAGS.batch_size],
          labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
          keep_prob: 0.5})

      train_accuracy = sess.run(acc, feed_dict={
        images_placeholder: train_image,
        labels_placeholder: train_label,
        keep_prob: 1.0})
      print ("step %d, training accuracy %g"%(step, train_accuracy))
      print ("loss:" + str(loss_value) + "test accuracy %g"%sess.run(acc, feed_dict={
        images_placeholder: test_image,
        labels_placeholder: test_label,
        keep_prob: 1.0}))

      summary_str = sess.run(summary_op, feed_dict={
        images_placeholder: train_image,
        labels_placeholder: train_label,
        keep_prob: 1.0})
      summary_writer.add_summary(summary_str, step)

  print ("test accuracy %g"%sess.run(acc, feed_dict={
    images_placeholder: test_image,
    labels_placeholder: test_label,
    keep_prob: 1.0}))
  save_path = saver.save(sess, "model.ckpt")