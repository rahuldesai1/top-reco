import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import keras
import numpy as np

def run(model, unnormalized_data_path, output_path):
	print("[PROCESSING] Loading data")
	normalizer = pd.read_csv("dependencies/scaler_data.csv", delimiter=',')
	data = pd.read_csv(unnormalized_data_path, delimiter=',')
	print("[INFO] Done loading data")

def normalize(col):
    norm_col = normalizer.loc[normalizer["column_name"] == str(col.name)]
    threshold = norm_col["threshold"].values[0]
    mini = norm_col["minimum"].values[0]
    slopeUpper = norm_col["slope_upper"].values[0]
    slopeLower = norm_col["slope_lower"].values[0]
    def norm_helper(row):
        if row > threshold:
            return 0.9 + slopeUpper * (row - threshold)
        else:
            return 0 + slopeLower * (row - mini)
    return col.apply(norm_helper)

#data = pd.read_csv('~/projects/samples/norm_results.csv', delimiter=",")
X_test = data.sample(frac=0.3)
y = X_test['label']
#X_norm = X_test.drop("label", axis=1).drop("rndm", axis=1, errors='ignore').drop("weight", axis=1).drop('Unnamed: 0', axis=1, errors='ignore')
X_test = X_test.drop('label', axis=1).drop("rndm", axis=1).drop("weight", axis=1).drop("Unnamed: 43", axis=1)
X_norm = X_test.apply(normalize, axis=0)
print(X_norm.head())


sess = tf.Session()
saver = tf.train.import_meta_graph("searched_models/model_1/model.meta")
saver.restore(sess, "searched_models/model_1/model")
print("Model Restored")

graph = tf.get_default_graph()

X_tr = graph.get_tensor_by_name('X_tr:0')
y_tr = graph.get_tensor_by_name('y_tr:0')
thresh = graph.get_tensor_by_name('thresh:0')

prob = graph.get_tensor_by_name("Probabilities:0")

#for op in graph.get_operations():
   # print(str(op.name))

probab = sess.run(prob, feed_dict={X_tr: X_norm.values, y_tr: tf.keras.utils.to_categorical(y), thresh: 0.5})
print(probab)
for t in np.arange(0, 1, 0.01):
    prediction = np.multiply((probab >= t)[:, 1], 1)

    TP = np.count_nonzero(prediction * y)
    TN = np.count_nonzero((prediction - 1) * (y - 1))
    FP = np.count_nonzero(prediction * (y - 1))
    FN = np.count_nonzero((prediction - 1) * y)
    
    signalEff = TP / (TP + FN)
    backgroundAccept = FP / (TN + FP)
    print(str(signalEff) + ", " + str(backgroundAccept))


