import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import keras
import sys

model_number = sys.argv[1]
print("Model Number: {0}".format(model_number))
# Use header to load my csv file with variable names
header = ['label','rndm','weight']
# boosted frame
header += ['W_B_pt','W_B_eta','W_B_phi','W_B_E','W_B_m']
header += ['b_B_pt','b_B_eta','b_B_phi','b_B_E','b_B_m']
header += ['Wjj_B_dR','tWb_B_dR'] #dR = sqrt(phi^2 + eta^2)
header += ['Wjj_B_deta','Wjj_B_dphi','tWb_B_deta','tWb_B_dphi']
# lab frame
header += ['W_pt','W_eta','W_phi','W_E','W_m']
header += ['b_pt','b_eta','b_phi','b_E','b_m']
header += ['Wjj_dR','tWb_dR']
header += ['Wjj_deta','Wjj_dphi','tWb_deta','tWb_dphi']
header += ['t_pt','t_eta','t_phi','t_E','t_m']

header += ['btag1', 'btag2', 'btag3'] #binary representation of likelihood of the jet being an actual bjet

#df = pd.read_csv('~/projects/top-reco-tests/samples/result.csv', names=header, delimiter=' ', skiprows=1)
df = pd.read_csv('~/projects/top-reco-tests/samples/norm_results_0.1.csv', delimiter=',')

#down-sample the class of non-jet samples to 1/4 of the original size (prevents model bias towards to majority class)
#pos_class = df[df['label'] == 1]
#neg_class = df[df['label'] == 0]
#neg_class = neg_class.sample(frac=0.1)
#neg_class.shape
#final_df = pd.concat([neg_class, pos_class])

#dataframe preprocessing
y = df['label']
X_norm = df.drop('label', axis=1).drop('Unnamed: 0', axis=1)

#data normalization
def normalize(col):
    print(col.name)
    threshold = col.quantile(0.9)
    mini = col.min()
    slopeUpper = (1 - 0.9) / (col.max() - threshold)
    slopeLower = (0.6 - 0) / (threshold - mini)
    def norm_helper(row):
        if row > threshold:
            return 0.9 + slopeUpper * (row - threshold)
        else:
            return 0 + slopeLower * (row - mini)
    return col.apply(norm_helper)

#normalize the data 
#X_norm = X.apply(normalize, axis=0)

print(X_norm.head())

#Randomly split the data for training and testing using sklearn's train_test_split method
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=.15, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.15, shuffle=True)

#print out the number of events for the trianing, testing, and validation sets
print("X_train events: " + str(X_train.shape[0]))
print("X_test events: " + str(X_test.shape[0]))
print("X_val events: " + str(X_val.shape[0]))
print("Pos training events: " + str(sum(y_train == 1)))
print("Negative training events: " + str(sum(y_train == 0)))

#calculate the % of samples that are positively labeled in the training data
print("Percent of positive class in the training set: {0}".format(sum(y_train == 1) / (sum(y_train == 1) + sum(y_train == 0))))

#Hyper-parameters
max_num_epochs = 10
batch_size = 256
learning_rate = 0.01
num_batches = int(len(X_train) / batch_size)

#Network Paramters
n_features = X_train.shape[1]
n_nodes_1 = 64
n_nodes_2 = 64
n_nodes_3 = 64
n_nodes_4 = 64
n_nodes_5 = 64
n_nodes_6 = 64
n_nodes_7 = 32
n_out = 2

#Create placeholders for the input data
X_tr = tf.placeholder(tf.float32, shape=[None, n_features])
y_tr = tf.placeholder(tf.float32, shape=[None, 2])

#Build the graph
W1 =tf.Variable(tf.truncated_normal([n_features, n_nodes_1], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[1, n_nodes_1]))
layer1 = tf.nn.relu(tf.matmul(X_tr, W1) + b1)

W2 =tf.Variable(tf.truncated_normal([n_nodes_1, n_nodes_2], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[1, n_nodes_2]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

W3 =tf.Variable(tf.truncated_normal([n_nodes_2, n_nodes_3], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[1, n_nodes_3]))
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

W4 =tf.Variable(tf.truncated_normal([n_nodes_3, n_nodes_4], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, shape=[1, n_nodes_4]))
layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)

W5 = tf.Variable(tf.truncated_normal([n_nodes_4, n_nodes_5], stddev=0.1))
b5 = tf.Variable(tf.constant(0.1, shape=[1, n_nodes_5]))
layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)

W6 =tf.Variable(tf.truncated_normal([n_nodes_5, n_nodes_6], stddev=0.1))
b6 = tf.Variable(tf.constant(0.1, shape=[1, n_nodes_6]))
layer6 = tf.nn.relu(tf.matmul(layer5, W6) + b6)

W7 =tf.Variable(tf.truncated_normal([n_nodes_6, n_nodes_7], stddev=0.1))
b7 = tf.Variable(tf.constant(0.1, shape=[1, n_nodes_7]))
layer7 = tf.nn.relu(tf.matmul(layer6, W7) + b7)

W_out = tf.Variable(tf.truncated_normal([n_nodes_7, n_out], stddev=0.1))
b_out = tf.Variable(tf.constant(0.1, shape=[1, n_out]))
logits = tf.matmul(layer7, W_out) + b_out

#Calculate the loss using Cross Entropy and define the optimizer
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_tr)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#compute the accuracy using Area Under Curve -- this gives us a better metric for imbalanced datasets
probabilities = tf.nn.softmax(logits)
prediction = tf.cast(tf.argmax(probabilities, axis=1), tf.float32)
actual = tf.cast(tf.argmax(y_tr, axis=1), tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, actual), tf.float32))

#create the session
sess = tf.InteractiveSession()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
saver = tf.train.Saver()

#run the model on the input data and calculate/print validation and training accuracy for each epoch
epoch = 0
best_val_acc = 0
acc_decreased = False
acc_decrease_threshold = 0.1

while epoch < max_num_epochs:
    total_loss = 0
    for batch in range(num_batches):
        index = batch * batch_size
        last = index + batch_size
        #training
        X_batch, y_batch = X_train.iloc[index:last].values, tf.keras.utils.to_categorical(y_train.iloc[index:last])
        _, batch_loss, acc = sess.run([optimizer, loss, accuracy], feed_dict={X_tr: X_batch, y_tr: y_batch})
        total_loss += batch_loss
    print("Epoch {0} ==> Acc:{1}, Loss: {2}".format(epoch, acc, total_loss))
    #test validation accuracy every 10 epochs
    if epoch % 5 == 0 and epoch != 0:
        val_acc = sess.run([accuracy], feed_dict={X_tr: X_val.values, y_tr: tf.keras.utils.to_categorical(y_val)})
        print("Val Accuracy: {0}".format(val_acc))
        
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
test_acc, predictions = sess.run([accuracy, prediction], feed_dict={X_tr: X_test.values, y_tr: tf.keras.utils.to_categorical(y_test)})
print("Test Accuracy: {0}".format(test_acc))


#Print out the percent of values that were classified as tripets in the test set 
print("Percent classified as a triplet: {0}".format(sum(predictions) / X_test.shape[0]))
sum(predictions)

#save the model
save_path = saver.save(sess, "models/model{0}.ckpt".format(model_number))
print("Model saved in path: %s" % save_path)

#saver.restore(sess, "models/model.ckpt")
#print("Model Restored")

#close the current session and drop all saved variable values. --MAKE SURE TO SAVE FIRST--
sess.close()
