import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from densenet import densenet
import config as cfg

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
batch_images = tf.reshape(x, [-1, 28, 28, 1])

label = tf.placeholder(tf.float32, shape=[None, 10])

training_flag = tf.placeholder(tf.bool)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = densenet.DenseNet(x=batch_images, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

"""
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)
In paper, use MomentumOptimizer
init_learning_rate = 0.1
but, I'll use AdamOptimizer
"""

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=cfg.epsilon)
train = optimizer.minimize(cost)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)

    global_step = 0
    epoch_learning_rate = cfg.init_learning_rate
    for epoch in range(cfg.total_epochs):
        if epoch == (cfg.total_epochs * 0.5) or epoch == (cfg.total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        total_batch = int(mnist.train.num_examples / cfg.batch_size)

        for step in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(cfg.batch_size)

            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }

            _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

            if step % 100 == 0:
                global_step += 100
                train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
                # accuracy.eval(feed_dict=feed_dict)
                print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
                writer.add_summary(train_summary, global_step=epoch)

            test_feed_dict = {
                x: mnist.test.images,
                label: mnist.test.labels,
                learning_rate: epoch_learning_rate,
                training_flag: False
            }

        accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
        print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates)
        # writer.add_summary(test_summary, global_step=epoch)

    saver.save(sess=sess, save_path='./model/dense.ckpt')