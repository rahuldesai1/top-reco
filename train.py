import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import keras
import sys

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
y = df['label']
X_norm = df.drop('label', axis=1).drop('Unnamed: 0', axis=1)

#Randomly split the data for training and testing using sklearn's train_test_split method
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=.15, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.15, shuffle=True)

#calculate the % of samples that are positively labeled in the training data
print("Percent of positive class in the training set: {0}".format(sum(y_train == 1) / (sum(y_train == 1) + sum(y_train == 0))))

#Hyper-parameters
max_num_epochs = 35
num_batches = int(len(X_train) / batch_size)

#Network Paramters
n_features = X_train.shape[1]
layer_nodes = []
for _ in range(number_layers - 1):
    layer_nodes.append(64)
layer_nodes.append(32)
n_out = 2

#Create placeholders for the input data
X_tr = tf.placeholder(tf.float32, shape=[None, n_features])
y_tr = tf.placeholder(tf.float32, shape=[None, 2])

#Build the graph
layers = []

for _ in range(number_layers):
    layers.append(0)

def make_hidden(n_nodes_in, n_nodes_out):
    W = tf.Variable(tf.truncated_normal([n_nodes_in, n_nodes_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[1, n_nodes_out]))
    return {'weight': W, 'bias': b}

#Create placeholders for the input data
X_tr = tf.placeholder(tf.float32, shape=[None, n_features])
y_tr = tf.placeholder(tf.float32, shape=[None, 2])

layer_1 = make_hidden(n_features, layer_nodes[0])
layer_1 = tf.nn.relu(tf.matmul(X_tr, layer_1['weight']) + layer_1['bias'])
layers[0] = tf.layers.dropout(layer_1, dropout)

for i in range(1, number_layers):
    hidden_layer = make_hidden(layer_nodes[i - 1], layer_nodes[i])
    layers[i] = tf.nn.relu(tf.matmul(layers[i - 1], hidden_layer['weight']) + hidden_layer['bias'])

W_out = make_hidden(layer_nodes[-1], n_out)
logits = tf.matmul(layers[-1], W_out['weight']) + W_out['bias']

if loss_func == "weighted":
    entropy = tf.nn.weighted_cross_entropy_with_logits(targets=y_tr, logits=logits, pos_weight=2.0)
else:
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_tr, logits=logits)
loss = tf.reduce_mean(entropy)
optimizer = optim(learning_rate=learning_rate).minimize(loss)

#compute the accuracy using Area Under Curve -- this gives us a better metric for imbalanced datasets
probabilities = tf.nn.softmax(logits)
prediction = tf.cast(tf.argmax(probabilities, axis=1), tf.float32)
actual = tf.cast(tf.argmax(y_tr, axis=1), tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, actual), tf.float32))
precision = tf.metrics.precision(labels=actual, predictions=prediction)
recall = tf.metrics.recall(labels=actual, predictions=prediction)

#create the session
sess = tf.InteractiveSession()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
saver = tf.train.Saver()

#run the model on the input data and calculate/print validation and training accuracy for each epoch
epoch = 0
best_val_acc = 0
acc_decreased = False
acc_decrease_threshold = 0.1
training_accuracy = []

while epoch < max_num_epochs:
    total_loss = 0
    total_acc = 0
    total_prec = 0
    total_recc = 0
    for batch in range(num_batches):
        index = batch * batch_size
        last = index + batch_size
        #training
        X_batch, y_batch = X_train.iloc[index:last].values, tf.keras.utils.to_categorical(y_train.iloc[index:last])
        _, batch_loss, acc, prec, recc = sess.run([optimizer, loss, accuracy, precision, recall], feed_dict={X_tr: X_batch, y_tr: y_batch})
        total_loss += batch_loss
        total_acc += acc
        total_prec += prec[0]
        total_recc += recc[0]
    training_accuracy.append(total_acc / num_batches)
    print("Epoch {0} ==> Acc: {1}, Precision: {2}, Recall: {3}, Loss: {4}".format(epoch, total_acc / num_batches, total_prec/float(num_batches), total_recc/float(num_batches), total_loss))
    #test validation accuracy every 10 epochs
    if epoch % 5 == 0 and epoch != 0:
        val_acc, prec, recc = sess.run([accuracy, precision, recall], feed_dict={X_tr: X_val.values, y_tr: tf.keras.utils.to_categorical(y_val)})
        print("Val Accuracy: {0}, Precision: {1}, Recall: {2}".format(float(val_acc), prec, recc))
        
        #implement early stopping
        if (best_val_acc > (float(val_acc) * (1 + acc_decrease_threshold))):
            if (acc_decreased):
                break
            else:
                acc_decreased = True
        else:
            best_val_acc = max(float(val_acc), best_val_acc)
            acc_decreased = False
        
    epoch += 1;

#test the model on test data that was take from result.csv 
test_acc, predictions, prec, recc, prob = sess.run([accuracy, prediction, precision, recall, probabilities], feed_dict={X_tr: X_test.values, y_tr: tf.keras.utils.to_categorical(y_test)})
print("Test Accuracy: {0}, Precision: {1}, Recall: {2}".format(test_acc, prec, recc))

#Print out the percent of values that were classified as tripets in the test set 
print("Percent classified as a triplet: {0}".format(sum(predictions) / X_test.shape[0]))
print(sum(predictions))

model_number = "SF:{0}_layers{1}_BS{2}_LR{3}_OP{4}_LS{5}_DO{6}".format(sample_frac, number_layers, batch_size, learning_rate, optim, loss_func, dropout)
#save the model
try:
    os.makedirs("searched_models/{0}/".format(model_number)) 
except FileExistsError:
    pass

train_accuracy = "Train Accuracy: " + str(train_accuracy)
test_accuracy = "Test Accuracy: " + str(test_acc)
with open("searched_models/{0}/model_evaluation.txt".format(model_number), "w") as file:
    file.writelines([train_accuracy, test_accuracy, "Training Labels: " + str(y_tr), "Output Probabilities: " + str(prob)])
    
save_path = saver.save(sess, "searched_models/{0}/model.ckpt".format(model_number))
print("Model saved in path: %s" % save_path)

#close the current session and drop all saved variable values. --MAKE SURE TO SAVE FIRST--
sess.close()
