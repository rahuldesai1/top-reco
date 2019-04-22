import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import keras

print("Loading Data...")
normalizer = pd.read_csv("~/projects/samples/scaler_data.csv", delimiter=',')
data = pd.read_csv('~/projects/samples/result.csv', delimiter=' ')
print("Done!")

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

X_test = data.sample(frac=0.2)
y = X_test['label']
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

prob = graph.get_operation_by_name("Probabilities:0")
prediction = tf.cast(np.multiply((prob >= thresh)[:, 1], 1), tf.float32)
actual = tf.cast(tf.argmax(y_tr, axis=1), tf.float32, name="groundTruth")
TP = tf.count_nonzero(prediction * actual)
TN = tf.count_nonzero((prediction - 1) * (actual - 1))
FP = tf.count_nonzero(prediction * (actual - 1))
FN = tf.count_nonzero((prediction - 1) * actual)
signalEff = TP / (TP + FN)
backgroundAccept = FP / (TN + FP)


sess.run(tf.global_variables_initializer())
prec, recc = sess.run([signalEff, backgroundAccept], feed_dict={X_tr: X_norm.values, y_tr: tf.keras.utils.to_categorical(y), thresh: 0.5})
print("SE: {0}, BA: {1}".format(prec, recc))




