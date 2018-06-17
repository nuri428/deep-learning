import tensorflow as tf
c = []

for d in ['/gpu:0', '/gpu:1']:
	with tf.device(d):
		a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2,3], name='a')
		b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3,2], name='b')
		c.append(tf.matmul(a,b))
	
with tf.device("/cpu:0"):
	sum = tf.add_n(c)
	
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print sess.run(sum)
