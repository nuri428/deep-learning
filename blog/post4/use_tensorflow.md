# tensorflow 학습 시킨 모델 사용하기 

```python
# Construct model
pred = somemodel.buildmodel(
    x, keep_prob, n_classes, imagesize, img_channel)

# Define loss and optimizer
 loss = tf.reduce_mean(-(y * tf.log(pred + 1e-12) + (1 - y) * tf.log(1 - pred + 1e-12)))
 cross_entropy = tf.reduce_mean(tf.reduce_sum(loss, reduction_indices=[1]))
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learning_rate, global_step, 1000, decay_rate, staircase=True)
# lr = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
    cross_entropy, global_step=global_step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.to_float(correct_pred))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 1
    writer = tf.train.SummaryWriter("./../logs", sess.graph)

    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        train_image, train_label = dataset.train.next_batch(
            batch_size=batch_size)
        summary, _ = sess.run([merged, optimizer], feed_dict={
            x: train_image, y: train_label, keep_prob: dropout})

        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={
                           x: train_image, y: train_label, keep_prob: 1.})
            # Calculate batch loss
            cVal = sess.run(cross_entropy, feed_dict={
                x: train_image, y: train_label, keep_prob: 1.})

            print "Iter " + str(step) + ", Minibatch Loss= " + "{}".format(cVal) + ", Training Accuracy= " + "{}".format(acc)
            # print "Testing Accuracy:", sess.run([pred], feed_dict={x:
            # dataset.test.images[0:3], keep_prob: 1.})

        step += 1

    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: dataset.test.images, y: dataset.test.labels, keep_prob: 1.})

# 여기까지가 학습.. 

    
# 여기서 부터 학습된 모델을 사용 

    print "predict test  class:", sess.run([pred], feed_dict={x: dataset.test.images[0:3], keep_prob: 1.})[0]

    print "predict valid class:", sess.run([pred], feed_dict={x: dataset.valid.images, keep_prob: 1.})[0]*100

```