#!/usr/bin/env python3

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import keras
import sys
import os
from operator import itemgetter

def train(data, output_dir, max_num_epochs, number_layers, batch_size, optim, learning_rate, dropout):
	optim = eval(optim)
	#dataframe preprocessing
	y = data['label']
	print(y)
	X_norm = data.drop('label', axis=1)
	print(X_norm.head())

	#Randomly split the data for training and testing using sklearn's train_test_split method
	X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=.15, shuffle=True)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.15, shuffle=True)

	#calculate the % of samples that are positively labeled in the training data
	print("Percent of positive class in the training set: {0}".format(sum(y_train == 1) / (sum(y_train == 1) + sum(y_train == 0))))

	#Hyper-parameters
	num_batches = int(len(X_train) / batch_size)
	print(len(X_train))
	print(batch_size)
	print(num_batches)

	#Network Paramters
	n_features = X_train.shape[1]
	layer_nodes = []
	for _ in range(number_layers - 1):
    		layer_nodes.append(64)
	layer_nodes.append(32)
	n_out = 2

	#Create placeholders for the input data
	X_tr = tf.placeholder(tf.float32, shape=[None, n_features], name="X_tr")
	y_tr = tf.placeholder(tf.float32, shape=[None, 2], name="y_tr")
	threshold = tf.placeholder(tf.float32, shape=(), name="thresh")
	drop_prob = tf.placeholder_with_default(1.0, shape=())

	#Build the graph
	layers = []

	for _ in range(number_layers):
    		layers.append(0)

	def make_hidden(n_nodes_in, n_nodes_out, num):
		W = tf.Variable(tf.truncated_normal([n_nodes_in, n_nodes_out], stddev=0.1), name="W"+num)
		b = tf.Variable(tf.constant(0.1, shape=[1, n_nodes_out]), name="b"+num)
		return {'weight': W, 'bias': b}

	#Create placeholders for the input data
	layer_1 = make_hidden(n_features, layer_nodes[0], "0")
	layers[0] = tf.nn.relu(tf.matmul(X_tr, layer_1['weight']) + layer_1['bias'])

	for i in range(1, number_layers - 1):
		hidden_layer = make_hidden(layer_nodes[i - 1], layer_nodes[i], str(i))
		active_layer = tf.nn.relu(tf.matmul(layers[i - 1], hidden_layer['weight']) + hidden_layer['bias'])
		layers[i] = tf.layers.dropout(active_layer, drop_prob)

	hidden_last = make_hidden(layer_nodes[number_layers - 2], layer_nodes[number_layers - 1], str(number_layers - 1))
	layers[number_layers - 1] = tf.nn.relu(tf.matmul(layers[number_layers - 2], hidden_last['weight']) + hidden_last['bias'])

	W_out = make_hidden(layer_nodes[-1], n_out, str(number_layers))
	logits = tf.matmul(layers[-1], W_out['weight']) + W_out['bias']

	class_weights = tf.constant([0.1, 1.0])
	weighted_logits = tf.multiply(logits, class_weights)
	entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_tr, logits=weighted_logits)
	loss = tf.reduce_mean(entropy, name="Loss")

	optimizer = optim(learning_rate=learning_rate, name="Optimizer").minimize(loss)

	#compute the accuracy using Area Under Curve -- this gives us a better metric for imbalanced datasets
	probabilities = tf.nn.softmax(logits, name="Probabilities")
	prediction = tf.cast(tf.argmax(probabilities, axis=1), tf.float32, name="Predictions")
	actual = tf.cast(tf.argmax(y_tr, axis=1), tf.float32, name="groundTruth")

	TP = tf.count_nonzero(prediction * actual)
	TN = tf.count_nonzero((prediction - 1) * (actual - 1))
	FP = tf.count_nonzero(prediction * (actual - 1))
	FN = tf.count_nonzero((prediction - 1) * actual)
	signalEff = TP / (TP + FN)
	backgroundAccept = FP / (TN + FP)

	#create the session and initialize variables
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	#run the model on the input data and calculate/print validation and training accuracy for each epoch
	epoch = 0
	highest_se = 0
	se_decreased = False
	training_se = []
	training_ba = []

	while epoch < max_num_epochs:
		total_loss = 0
		total_se = 0
		total_ba = 0
		shuffled = pd.concat([X_train, y_train], axis=1)
		shuffled = shuffled.sample(frac=1)
		y_train = shuffled['label']
		X_train = shuffled.drop('label', axis=1)

		for batch in range(num_batches):
			index = batch * batch_size
			last = index + batch_size
        		#training
			X_batch, y_batch = X_train.iloc[index:last].values, tf.keras.utils.to_categorical(y_train.iloc[index:last])
			_, batch_loss, signal, back = sess.run([optimizer, loss, signalEff, backgroundAccept], feed_dict={X_tr: X_batch, y_tr: y_batch, threshold: 0.5, drop_prob: dropout})
			total_loss += batch_loss
			total_se += signal
			total_ba += back
		training_se.append(total_se / float(num_batches))
		training_ba.append(total_ba / float(num_batches))
		print("Epoch {0} ==> Signal Efficiency: {1}, Background Acceptance: {2}, Loss: {3}".format(epoch, total_se / num_batches, total_ba / num_batches, total_loss))
		#test validation accuracy every 10 epochs
		if epoch % 3 == 0 and epoch != 0:
			val_se, val_ba  = sess.run([signalEff, backgroundAccept], feed_dict={X_tr: X_val.values, y_tr: tf.keras.utils.to_categorical(y_val), threshold: 0.5, drop_prob:0.0})
			print("Validation Signal Efficiency: {0}, Validation Background Acceptance: {1}".format(val_se, val_ba))
        #implement early stopping
			if (highest_se > (float(val_se))):
				if (se_decreased):
					break
				else:
					se_decreased = True
			else:
				highest_se = val_se
				se_decreased = False
		epoch += 1;

	test_thresh = []
	#test the model on test data that was take from result.csv
	predictions, test_se, test_ba, prob = sess.run([prediction, signalEff, backgroundAccept, probabilities], feed_dict={X_tr: X_test.values, y_tr: tf.keras.utils.to_categorical(y_test), threshold: 0.5, drop_prob:0.0})
	print("[INFO] Test Signal Efficiency: {0}, Test Background Acceptance: {1}".format(test_se, test_ba))

	print("[PROCESSING] Testing Various Threshold Values")

	save_df = pd.DataFrame(columns=["Train_Results", "Test_Results", "Threshold Values"])
	final_dict = {}
	final_dict["Threshold Values"] = []
	for t in np.arange(0, 1, 0.01):
		temp_dict = {}
		temp_dict["Threshold"] = t
		prediction = np.multiply((prob >= t)[:, 1], 1)
		TP = np.count_nonzero(prediction * y_test)
		TN = np.count_nonzero((prediction - 1) * (y_test - 1))
		FP = np.count_nonzero(prediction * (y_test - 1))
		FN = np.count_nonzero((prediction - 1) * y_test)
		signalEff = TP / (TP + FN)
		backgroundAccept = FP / (TN + FP)
		temp_dict["Background Acceptance"] = backgroundAccept
		temp_dict["Signal Efficiency"] = signalEff
		final_dict["Threshold Values"].append(temp_dict)

	#save the model
	try:
		os.makedirs(output_dir)
	except FileExistsError:
		pass

	training_accuracy = list(zip(training_se, training_ba))
	final_dict["Train_Results"] = []
	count = 0

	for i in training_accuracy:
		new_dict = {}
		new_dict["Epoch"] = count
		new_dict["Metrics"] = []
		temp = {}
		temp["Signal Efficiency"] = i[0]
		temp["Background Acceptance"] = i[1]
		new_dict["Metrics"].append(temp)
		final_dict["Train_Results"].append(new_dict)
		count += 1
	final_dict["Test_Results"] = []
	final_dict["Test_Results"].append({"Signal Efficiency": test_se, "Background Acceptance": test_ba})

	save_df = save_df.append(final_dict, ignore_index=True)
	save_df.to_json("{0}/model_metrics.json".format(output_dir))

	df_pred = pd.DataFrame(data={'actual': y_test.values, 'neg_predictions': list(map(itemgetter(0), prob)), 'pos_predictions': list(map(itemgetter(1), prob))}).to_string()

	with open("{0}/model_results.txt".format(output_dir), "w+") as f:
		print("[INFO] Writing model metrics to model_results.txt")
		f.writelines([df_pred])

	save_path = saver.save(sess, "{0}/model".format(output_dir))
	print("[INFO] Model saved in path: %s" % save_path)

	#close the current session and drop all saved variable values. --MAKE SURE TO SAVE FIRST--
	sess.close()
