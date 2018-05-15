import tensorflow as tf
import numpy as np
import load_data as ld

from tensorflow.examples.tutorials.mnist import input_data

# signX = np.load('X.npy')
# signY = np.load('Y.npy')

learning_rate = 0.01
training_epochs = 15
batch_size = 100
size = 2062

(signX, signY) = ld.load_signlang(size)

X = tf.placeholder(tf.float32, [None, 64, 64])
X_img = tf.reshape(X, [-1, 64, 64, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 64, 64, 1)
W1= tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# Conv -> (?, 64, 64 ,32)
# Pool -> (?, 32, 32, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# L2 ImgIn shape=(?, 32, 32, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
# Conv -> (?, 32, 32, 64)
# Pool -> (?, 16, 16, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.reshape(L2, [-1, 16 * 16 * 64])

L2 = tf.reshape(L2, [-1, 16 * 16 * 64])

# Final FC 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W2", shape=[16 * 16 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b

# Define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train model
print('Learning Started!')
# feed_dict = {X: signX, Y: signY}
# c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(2062 / batch_size)
    i = 0
    while i <= total_batch:
        batch_xs = signX[i:i+batch_size]
        batch_ys = signY[i:i+batch_size]
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        i += 100
    print('Epoch:', '%04d' % (epoch + 1), 'cost:', '{:.9f}'.format(avg_cost))

# for epoch in range(training_epochs):
#     avg_cost = 0
    # total_batch = int(mnist.train.num_examples / batch_size)
    # for i in range(total_batch):
    #     batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #     feed_dict = {X: batch_xs, Y: batch_ys}
    #     c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    #     avg_cost += c / total_batch
    # print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: signX, Y: signY}))