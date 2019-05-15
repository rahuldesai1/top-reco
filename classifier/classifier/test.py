import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import keras
import numpy as np

def run(model_path, data, output_path):
	y = data['label']
	X_norm = data.drop('label', axis=1)

	print("[PROCESSING] Loading model from {0}".format(model_path))
	sess = tf.Session()
	saver = tf.train.import_meta_graph("{0}/model.meta".format(model_path))
	saver.restore(sess, "{0}/model".format(model_path))
	print("[INFO] Model Restored")

	graph = tf.get_default_graph()

	X_tr = graph.get_tensor_by_name('X_tr:0')
	y_tr = graph.get_tensor_by_name('y_tr:0')
	thresh = graph.get_tensor_by_name('thresh:0')
	drop_prob = graph.get_tensor_by_name('drop_prob:0')

	prob = graph.get_tensor_by_name("Probabilities:0")

	print("[PROCESSING] Running data through classifier")
	probab = sess.run(prob, feed_dict={X_tr: X_norm.values, y_tr: tf.keras.utils.to_categorical(y), thresh: 0.5, drop_prob:0.0})

	final_dict = {}
	final_dict["Threshold Values"] = []
	for t in np.arange(0, 1, 0.01):
		temp_dict = {}
		temp_dict["Threshold"] = t
		prediction = np.multiply((probab >= t)[:, 1], 1)
		TP = np.count_nonzero(prediction * y)
		TN = np.count_nonzero((prediction - 1) * (y - 1))
		FP = np.count_nonzero(prediction * (y - 1))
		FN = np.count_nonzero((prediction - 1) * y)
		signalEff = TP / (TP + FN)
		backgroundAccept = FP / (TN + FP)

		if t % 0.1 == 0:
			print("Threshold: {0}".format(t))
			print("Signal Efficiency: " + str(signalEff) + ", " + " background Acceptance: " + str(backgroundAccept))

		temp_dict["Background Acceptance"] = backgroundAccept
		temp_dict["Signal Efficiency"] = signalEff
		final_dict["Threshold Values"].append(temp_dict)

	save_df = pd.DataFrame(columns=["Threshold Values"])
	save_df = save_df.append(final_dict, ignore_index=True)

	#save the model
	try:
		os.makedirs(output_path)
	except FileExistsError:
		pass

	save_df.to_json("{0}/testing_results.json".format(output_path))

	print("[INFO] Output saved to {0}/testing_results.json".format(output_path))

def classify_triplet(model_path, data, thresh):
	y = data['label']
	X_norm = data.drop('label', axis=1)

	print("[PROCESSING] Loading model from {0}".format(model_path))
	sess = tf.Session()
	saver = tf.train.import_meta_graph("{0}/model.meta".format(model_path))
	saver.restore(sess, "{0}/model".format(model_path))
	print("[INFO] Model Restored")

	graph = tf.get_default_graph()

	X_tr = graph.get_tensor_by_name('X_tr:0')
	y_tr = graph.get_tensor_by_name('y_tr:0')
	thresh = graph.get_tensor_by_name('thresh:0')

	prob = graph.get_tensor_by_name("Probabilities:0")

	print("[PROCESSING] Running data through classifier")
	probab = sess.run(prob, feed_dict={X_tr: X_norm.values, y_tr: tf.keras.utils.to_categorical(y), thresh: 0.5})

	if probab[0] >= thresh:
		return 1
	else:
		return 0

