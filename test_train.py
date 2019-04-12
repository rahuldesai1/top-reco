import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import keras
import sys
import os
from operator import itemgetter

sample_frac = float(sys.argv[1])
number_layers = int(sys.argv[2])
batch_size = int(sys.argv[3])
optim = eval(sys.argv[4])
learning_rate = float(sys.argv[5])
loss_func = sys.argv[6]
dropout = float(sys.argv[7])

df = pd.read_csv('~/projects/samples/norm_results.csv', delimiter=',')

#down-sample the class of non-jet samples to 1/4 of the original size (prevents model bias towards to majority class)
pos_class = df[df['label'] == 1]
neg_class = df[df['label'] == 0]
neg_class = neg_class.sample(frac=sample_frac)
neg_class.shape
final_df = pd.concat([neg_class, pos_class])

#dataframe preprocessing
y = final_df['label']
final_df = final_df.drop('weight', axis=1)
X_norm = final_df.drop('label', axis=1).drop('Unnamed: 0', axis=1)

#Randomly split the data for training and testing using sklearn's train_test_split method
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=.15, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.15, shuffle=True)

#calculate the % of samples that are positively labeled in the training data
print("Percent of positive class in the training set: {0}".format(sum(y_train == 1) / (sum(y_train == 1) + sum(y_train == 0))))

#Hyper-parameters
max_num_epochs = 2
num_batches = int(len(X_train) / batch_size)

#Network Paramters
n_features = X_train.shape[1]
layer_nodes = []
for _ in range(number_layers - 1):
    layer_nodes.append(64)
layer_nodes.append(32)
n_out = 2

#Create placeholders for the input data
calc_accuracy = tf.placeholder(tf.bool)

if not calc_accuracy:
    X_tr = tf.placeholder(tf.float32, shape=[None, n_features], name="X_tr")
    y_tr = tf.placeholder(tf.float32, shape=[None, 2], name="y_tr")
else:
    probabilities = tf.placeholder(tf.float32, shape=(None, 2))
threshold = tf.placeholder(tf.float32, shape=(), name="thresh")

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
    layers[i] = tf.layers.dropout(active_layer, dropout)

hidden_last = make_hidden(layer_nodes[number_layers - 2], layer_nodes[number_layers - 1], str(number_layers - 1))
layers[number_layers - 1] = tf.nn.relu(tf.matmul(layers[number_layers - 2], hidden_last['weight']) + hidden_last['bias'])

W_out = make_hidden(layer_nodes[-1], n_out, str(number_layers))
logits = tf.matmul(layers[-1], W_out['weight']) + W_out['bias']

if loss_func == "weighted":
    class_weights = tf.constant([0.1, 1.5])
    weighted_logits = tf.multiply(logits, class_weights)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_tr, logits=weighted_logits)
else:
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_tr, logits=logits)
loss = tf.reduce_mean(entropy, name="Loss")
optimizer = optim(learning_rate=learning_rate, name="Optimizer").minimize(loss)

#compute the accuracy using Area Under Curve -- this gives us a better metric for imbalanced datasets
probabilities = tf.nn.softmax(logits, name="Probabilities")
#prediction = tf.cast(tf.argmax(probabilities, axis=1), tf.float32, name="Predictions")
prediction = tf.cast(np.multiply((probabilities >= threshold)[:, 1], 1), tf.float32, name="Predictions")
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
        _, batch_loss, signal, back = sess.run([optimizer, loss, signalEff, backgroundAccept], feed_dict={calc_accuracy: False, X_tr: X_batch, y_tr: y_batch, threshold: 0.5})
        total_loss += batch_loss
        total_se += signal
        total_ba += back
    training_se.append(total_se / float(num_batches))
    training_ba.append(total_ba / float(num_batches))
    print("Epoch {0} ==> Signal Efficiency: {1}, Background Acceptance: {2}, Loss: {3}".format(epoch, total_se / num_batches, total_ba / num_batches, total_loss))
    #test validation accuracy every 10 epochs
    if epoch % 3 == 0 and epoch != 0:
        val_se, val_ba  = sess.run([signalEff, backgroundAccept], feed_dict={calc_accuracy: False, X_tr: X_val.values, y_tr: tf.keras.utils.to_categorical(y_val), threshold: 0.5})
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
predictions, test_se, test_ba, prob = sess.run([prediction, signalEff, backgroundAccept, probabilities], feed_dict={calc_accuracy: False, X_tr: X_test.values, y_tr: tf.keras.utils.to_categorical(y_test), threshold: 0.5})
print("Test Signal Efficiency: {0}, Test Background Acceptance: {1}".format(test_se, test_ba))

print("Testing Various Threshold Values")
for t in np.arange(0.1, 1, 0.1):
    test_se, test_ba = sess.run([signalEff, backgroundAccept], feed_dict={calc_accuracy: True, probabilities: prob, threshold: t})
    test_thresh.append((t, test_se, test_ba))

#Print out the percent of values that were classified as tripets in the test set 
print("Percent classified as a triplet: {0}".format(sum(predictions) / X_test.shape[0]))

model_number = "SIGNAL_EFF:{7}__SF:{0}_layers:{1}_BS:{2}_LR:{3}_OP:{4}_LS:{5}_DO:{6}".format(sample_frac, number_layers, batch_size, learning_rate, sys.argv[4][9:], loss_func, dropout, test_se)
#save the model
try:
    os.makedirs("~/projects/searched_models/{0}/".format(model_number)) 
except FileExistsError:
    pass

hypers = model_number + "\n"
training_accuracy = list(zip(training_se, training_ba))
train_eval = "Train (Signal Efficiency, Background Acceptance): " + str(training_accuracy) + "\n"
test_acc = test_thresh
test_eval = "Test (Threshold, Signal Efficiency, Background Acceptance): " + str(test_acc) + "\n"
df_pred = pd.DataFrame(data={'actual': y_test.values, 'neg_predictions': list(map(itemgetter(0), prob)), 'pos_predictions': list(map(itemgetter(1), prob))}).to_string()

with open("~/projects/searched_models/{0}/model_evaluation.txt".format(model_number), "w") as file:
    file.writelines([hypers, train_eval, test_eval, "Predictions: " + df_pred])
    
save_path = saver.save(sess, "searched_models/{0}/model".format(model_number))
print("Model saved in path: %s" % save_path)

#close the current session and drop all saved variable values. --MAKE SURE TO SAVE FIRST--
sess.close()
