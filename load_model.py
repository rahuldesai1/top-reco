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
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=.4, shuffle=True)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "models/model{0}.ckpt".format(model_number))
print("Model Restored")

print("Running Model")
prec, recc = sess.run([precision, recall], feed_dict={X_tr: X_test.values, y_tr: tf.keras.utils.to_categorical(y_test)})
print("Precision: {0}, Recall: {1}".format(prec, recall))




