import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
from plot_utils import demo_simple_grid
from config import *

# display parameter values
for arg_name, arg_value in vars(ARGS).items():
	print (str(arg_name) + " = " + str(arg_value))

import os
samples_dir = ARGS.SAMPLES_DIR
if not os.path.exists(samples_dir):
	os.makedirs(samples_dir)

import sys
logfile = ARGS.LOGFILE
sys.stdout = open(log_file, 'w')

rts_mnist = np.load(ARGS.DATA_FOLDER + 'RTS_mnist.npz')
X, y = rts_mnist['distorted_x'], rts_mnist['labels']

stn_arch = ARGS.STN_ARCH
classifier_arch = ARGS.CLASSIFIER_ARCH

if stn_arch == 'CNN':
	X = X.reshape(X.shape + (1, ))
	x = tf.placeholder(tf.float32, [None, 42, 42, 1])
else:
	X = X.reshape(X.shape[0], 42 * 42)
	x = tf.placeholder(tf.float32, [None, 42*42])

y = tf.placeholder(tf.float32, [None, 10])

X_train = X[:10000] 
y_train = y[:10000] 
X_valid = X[10000:11000] 
y_valid = y[10000:11000]

Y_train = dense_to_one_hot(y_train, n_classes=10)
Y_valid = dense_to_one_hot(y_valid, n_classes=10) 

if stn_arch == 'CNN':
	filter_size=3
	n_filters_1=16
	W_loc1 = weight_variable([filter_size, filter_size, 1, n_filters_1], name='W_loc1')
	b_loc1 = bias_variable([n_filters_1], name='b_loc1')
	loc_conv1 = tf.nn.relu(
					tf.nn.conv2d(
						input=x, 
						filter=W_loc1, 
						strides=[1, 1, 1, 1], 
						padding='VALID'
					) + b_loc1
				)
	loc_pool1 = tf.nn.max_pool(
					value=loc_conv1, 
					ksize=[1, 2 ,2 , 1], 
					strides=[1, 2, 2, 1], 
					padding='VALID'
				)
	filter_size=3
	n_filters_2=16
	W_loc2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2], name='W_loc2')
	b_loc2 = bias_variable([n_filters_2], name='b_loc2')
	loc_conv2 = tf.nn.relu(
					tf.nn.conv2d(
						input=loc_pool1,
						filter=W_loc2,
						strides=[1, 1, 1, 1],
						padding='VALID'
					) + b_loc2
				)
	loc_pool2 = tf.nn.max_pool(
					value=loc_conv2,
					ksize=[1, 2, 2, 1],
					strides=[1, 2, 2, 1],
					padding='VALID'
				)
	# Shape of loc_pool2 should be (batch_size, 11, 11, n_filters_2)
	loc_pool2_flat = tf.reshape(loc_pool2, [-1, 121*n_filters_2])
	W_loc3 = weight_variable([121*n_filters_2, 256], name='W_loc3')
	b_loc3 = bias_variable([256], name='b_loc3')
	h_loc1 = tf.matmul(loc_pool2_flat, W_loc3) + b_loc3
	W_loc4 = tf.Variable(initial_value=tf.zeros([256, 6], tf.float32), name='W_loc4')
	b_loc4 = tf.Variable(initial_value=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], name='b_loc4')
	h_loc2 = tf.matmul(h_loc1, W_loc4) + b_loc4
	out_size = (42, 42)
	h_trans = transformer(x, h_loc2, out_size)
	stn_weights=[W_loc1, W_loc2, W_loc3, W_loc4]
	stn_biases=[b_loc1, b_loc2, b_loc3, b_loc4]

else:
	# STN architecture is fully connected, 1764 -> 1024 -> 256 -> 6
	W_loc1 = weight_variable([42*42, 1024], name='W_loc1')
	b_loc1 = bias_variable([1024], name='b_loc1')
	h_loc1 = tf.matmul(x, W_loc1) + b_loc1
	W_loc2 = weight_variable([1024, 256], name='W_loc2')
	b_loc2 = bias_variable([256], name='b_loc2')
	h_loc2 = tf.matmul(h_loc1, W_loc2) + b_loc2
	W_loc3 = weight_variable(initial_value=tf.zeros([256, 6], tf.float32), name='W_loc3')
	b_loc3 = bias_variable(initial_value=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], name='b_loc3')
	h_loc3 = tf.matmul(h_loc2, W_loc3) + b_loc3

	x_tensor = tf.reshape(x, [-1, 42, 42, 1])
	out_size = (42, 42)
	h_trans = transformer(x_tensor, h_loc3, out_size)
	stn_weights=[W_loc1, W_loc2, W_loc3]
	stn_biases=[b_loc1, b_loc2, b_loc3]

if classifier_arch == 'CNN':
	filter_size=3
	n_filters_1=16
	W_clsfr_1 = weight_variable([filter_size, filter_size, 1, n_filters_1], name='W_clsfr_1')
	b_clsfr_1 = bias_variable([n_filters_1], name='b_clsfr_1')
	clsfr_conv1 = tf.nn.relu(
					tf.nn.conv2d(
						input=h_trans, 
						filter=W_clsfr_1, 
						strides=[1, 1, 1, 1], 
						padding='VALID'
					) + b_clsfr_1
				)
	clsfr_pool1 = tf.nn.max_pool(
					value=clsfr_conv1, 
					ksize=[1, 2 ,2 , 1], 
					strides=[1, 2, 2, 1], 
					padding='VALID'
				)
	filter_size=3
	n_filters_2=32
	W_clsfr_2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2], name='W_clsfr_2')
	b_clsfr_2 = bias_variable([n_filters_2], name='b_clsfr_2')
	clsfr_conv2 = tf.nn.relu(
					tf.nn.conv2d(
						input=clsfr_pool1,
						filter=W_clsfr_2,
						strides=[1, 1, 1, 1],
						padding='VALID'
					) + b_clsfr_2
				)
	clsfr_pool2 = tf.nn.max_pool(
					value=clsfr_conv2,
					ksize=[1, 2, 2, 1],
					strides=[1, 2, 2, 1],
					padding='VALID'
				)
	# Shape of clsfr_pool2 should be (batch_size, 11, 11, n_filters_2)
	clsfr_pool2_flat = tf.reshape(clsfr_pool2, [-1, 121*n_filters_2])
	W_clsfr_3 = weight_variable([121*n_filters_2, 1024], name='W_clsfr_3')
	b_clsfr_3 = bias_variable([1024], name='b_clsfr_3')
	h_clsfr_1 = tf.nn.relu(
					tf.matmul(clsfr_pool2_flat, W_clsfr_3) 
					+ b_clsfr_3
				)
	W_clsfr_4 = weight_variable([1024, 10], name='W_clsfr_4')
	b_clsfr_4 = bias_variable([10], name='b_clsfr_4')
	y_logits = tf.nn.relu(
					tf.matmul(h_clsfr_1, W_clsfr_4) 
					+ b_clsfr_4
				)
	clsfr_weights=[W_clsfr_1, W_clsfr_2, W_clsfr_3, W_clsfr_4]
	clsfr_biases=[b_clsfr_1, b_clsfr_2, b_clsfr_3, b_clsfr_4]

else:
	h_trans_flat = tf.reshape(h_trans, [-1, 42*42])
	W_clsfr_1 = weight_variable([42*42, 1024], name='W_clsfr_1')
	b_clsfr_1 = bias_variable([1024], name='b_clsfr_1')
	h_clsfr_1 = tf.nn.relu(
					tf.matmul(h_trans_flat, W_clsfr_1) + b_clsfr_1
				)
	W_clsfr_2 = weight_variable([1024, 256], name='W_clsfr_2')
	b_clsfr_2 = bias_variable([256], name='b_clsfr_2')
	h_clsfr_2 = tf.nn.relu(
					tf.matmul(h_clsfr_1, W_clsfr_2) + b_clsfr_2
				)
	W_clsfr_3 = weight_variable([256, 10], name='W_clsfr_3')
	b_clsfr_3 = bias_variable([10], name='b_clsfr_3')
	y_logits  = tf.nn.relu(
					tf.matmul(h_clsfr_2, W_clsfr_3) + b_clsfr_3
				)
	clsfr_weights=[W_clsfr_1, W_clsfr_2, W_clsfr_3]
	clsfr_biases=[b_clsfr_1, b_clsfr_2, b_clsfr_3]

beta = ARGS.BETA
if ARGS.REG == 'L1':
	regularizer = tf.contrib.layers.l1_regularizer(scale=beta, scope=None)
elif ARGS.REG == 'L2':
	regularizer = tf.contrib.layers.l2_regularizer(scale=beta, scope=None)

if ARGS.PRETRAINED:
	reg_weights=stn_weights
else:
	reg_weights=stn_weights + clsfr_weights

if ARGS.reg=='None':
	reg_penalty=0
else:
	reg_penalty=tf.contrib.layers.apply_regularization(regularizer, reg_weights)
 
cross_entropy = tf.reduce_mean(
    				tf.nn.softmax_cross_entropy_with_logits(y_logits, y)
    			)
opt = tf.train.AdamOptimizer(learning_rate=ARGS.LEARNING_RATE)

if ARGS.PRETRAINED:
	optimizer = opt.minimize(cross_entropy + reg_penalty, var_list=stn_weights+stn_biases)
else:
	optimizer = opt.minimize(cross_entropy + reg_penalty)

correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if ARGS.PRETRAINED:
	restore_layers = {}
	clsfr_params = clsfr_weights + clsfr_biases
	for param in clsfr_params:
		restore_layers[str(param)] = param
	saver = tf.train.Saver(restore_layers)

iter_per_epoch=100
n_epochs=ARGS.N_EPOCHS

indices=np.linspace(0, 10000-1, iter_per_epoch)
indices=indices.astype('int')

for epoch_i in range(n_epochs):
	for iter_i in range(iter_per_epoch-1):
		batch_xs=X_train[indices[iter_i]:indices[iter_i+1]]
		batch_ys=Y_train[indices[iter_i]:indices[iter_i+1]]

		if iter_i%10 == 0:
			loss = sess.run(
								cross_entropy,
								feed_dict = {
								x:batch_xs,
								y:batch_ys
								}
							)
            print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))

		sess.run(
					optimizer, 
					feed_dict={
						x:batch_xs,
						y:batch_ys
					}
				)
		gen_images = sess.run(
								h_trans, 
								feed_dict={
									x:batch_xs,
									y:batch_ys
								}
							)
		gen_images = np.squeeze(gen_images)
		demo_simple_grid(gen_images[:25], figname=samples_dir+"/epoch-%03d.png" %epoch_i)

		acc = str(sess.run(
							accuracy, 
							feed_dict={
								x:X_valid,
								y:Y_valid
							}
						))
    	print('Accuracy (%d): %s' % (epoch_i, acc))




	

