import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import keras

model_number = sys.argv[1]

print("Loading Data...")
data = pd.read_csv('~/projects/samples/norm_results.csv', delimiter=',')
print("Done!")

y = data['label']
X_norm = data.drop('label', axis=1).drop("Unnamed: 0", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=.3, shuffle=True)

sess = tf.Session()
saver = tf.train.import_meta_graph("searched_models/PRECISION:0.7034680247306824__SF:0.3_layers:8_BS:256_LR:0.01_OP:GradientDescentOptimizer_LS:un_weighted_DO:0.5/model.meta".format(model_number))
saver.restore(sess, "searched_models/PRECISION:0.7034680247306824__SF:0.3_layers:8_BS:256_LR:0.01_OP:GradientDescentOptimizer_LS:un_weighted_DO:0.5/model".format(model_number))
print("Model Restored")

graph = tf.get_default_graph()
X_tr = graph.get_tensor_by_name('X_tr:0')
y_tr = graph.get_tensor_by_name('y_tr:0')

prediction = graph.get_tensor_by_name('Predicitions:0')
actual = tf.cast(tf.argmax(y_tr, axis=1), tf.float32, name="groundTruth")
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, actual), tf.float32), name="Accuracy")
precision = tf.metrics.precision(labels=actual, predictions=prediction, name="Precision")
recall = tf.metrics.recall(labels=actual, predictions=prediction, name="Recall")

sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
prec, recc = sess.run([precision, recall], feed_dict={X_tr: X_test.values, y_tr: tf.keras.utils.to_categorical(y_test)})
print("Precision: {0}, Recall: {1}".format(prec, recc))




