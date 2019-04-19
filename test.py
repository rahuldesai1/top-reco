import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import keras

model_number = sys.argv[1]

print("Loading Data...")
data = pd.read_csv('~/projects/samples/result.csv', delimiter=' ')
print("Done!")
normalizer = pd.read_csv("~/projects/samples/scaler_data.csv", delimiter=',')

def normalize(col):
    print(col.name)
    norm_col = normalizer[normalizer["column_name" == col]]
    threshold = norm_col["threshold"]
    mini = norm_col["minimum"]
    slopeUpper = norm_col["slope_upper"]
    slopeLower = norm_col["slope_lower"]
    def norm_helper(row):
        if row > threshold:
            return 0.9 + slopeUpper * (row - threshold)
        else:
            return 0 + slopeLower * (row - mini)
    return col.apply(norm_helper)

X_test = data.sample(frac=0.3)
print(X_test.columns.values)
y = X_test['label']
X_test = X_test.drop('label', axis=1)
X_norm = X_test.apply(normalize, axis=0)


sess = tf.Session()
saver = tf.train.import_meta_graph("searched_models/SIGNAL_EFF:0.5712634894600488__SF:0.3_layers:9_BS:256_LR:0.01_OP:AdamOptimizer_LS:weighted_DO:0.5/model.meta".format(model_number))
saver.restore(sess, "searched_models/SIGNAL_EFF:0.5712634894600488__SF:0.3_layers:9_BS:256_LR:0.01_OP:AdamOptimizer_LS:weighted_DO:0.5/model".format(model_number))
print("Model Restored")

graph = tf.get_default_graph()
X_tr = graph.get_tensor_by_name('X_tr:0')
y_tr = graph.get_tensor_by_name('y_tr:0')
thresh = graph.get_tensor_by_name('thresh:0')

prediction = graph.get_tensor_by_name('Predicitions:0')
actual = tf.cast(tf.argmax(y_tr, axis=1), tf.float32, name="groundTruth")
TP = tf.count_nonzero(prediction * actual)
TN = tf.count_nonzero((prediction - 1) * (actual - 1))
FP = tf.count_nonzero(prediction * (actual - 1))
FN = tf.count_nonzero((prediction - 1) * actual)
signalEff = TP / (TP + FN)
backgroundAccept = FP / (TN + FP)


sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
prec, recc = sess.run([signalEff, backgroundAccept], feed_dict={X_tr: X_test.values, y_tr: tf.keras.utils.to_categorical(y), thresh: 0.3})
print("SE: {0}, BA: {1}".format(prec, recc))




