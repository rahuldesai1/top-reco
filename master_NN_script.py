import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import keras
import sys
import os
from multiprocessing import Pool

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

#dataframe preprocessing
y = df['label']
X_norm = df.drop('label', axis=1).drop('Unnamed: 0', axis=1)

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

def run(model_number, number_layers, layer_nodes, input_batch_size, optim, input_learning_rate, dropout):
    
    assert number_layers == len(layer_nodes), "Length of layer_nodes does not match size of network."    

    #Hyper-parameters
    max_num_epochs = 3
    batch_size = input_batch_size
    learning_rate = input_learning_rate
    num_batches = int(len(X_train) / batch_size)

    #Network Paramters
    n_features = X_train.shape[1]
    n_out = 2
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

    #Calculate the loss using Cross Entropy and define the optimizer
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_tr)
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

    while epoch < max_num_epochs:
        total_loss = 0
        total_acc = 0
        total_prec = 0
        for batch in range(num_batches):
	        index = batch * batch_size
	        last = index + batch_size
	        #training
	        X_batch, y_batch = X_train.iloc[index:last].values, tf.keras.utils.to_categorical(y_train.iloc[index:last])
	        _, batch_loss, acc, prec = sess.run([optimizer, loss, accuracy, precision], feed_dict={X_tr: X_batch, y_tr: y_batch})
	        total_loss += batch_loss
	        total_acc += acc
	        total_prec += prec[0]
        print("Epoch {0} ==> Acc: {1}, Precision: {2} , Loss: {3}".format(epoch, total_acc / num_batches, total_prec/float(num_batches),  total_loss))
	    #test validation accuracy every 10 epoch
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
    test_acc, predictions, prec, recc = sess.run([accuracy, prediction, precision, recall], feed_dict={X_tr: X_test.values, y_tr: tf.keras.utils.to_categorical(y_test)})
    print("Test Accuracy: {0}, Precision: {1}, Recall: {2}".format(test_acc, prec, recc))


    #Print out the percent of values that were classified as tripets in the test set 
    print("Percent classified as a triplet: {0}".format(sum(predictions) / X_test.shape[0]))

    #save the model
    try:
        os.makedirs("searched_models/model{0}/".format(model_number))
    except FileExistsError:
        pass
    save_path = saver.save(sess, "searched_models/model{0}/model{0}.ckpt".format(model_number))
    print("Model saved in path: %s" % save_path)

    #close the current session and drop all saved variable values. --MAKE SURE TO SAVE FIRST--
    sess.close()
    
    return (test_acc, prec)

def extract(rows):
	return True

if __name__ == '__main__':
    chunksize = 32
    for chunk in pd.read_csv('~/projects/hyperparameters.csv', chunksize=chunksize):
        #return the hyperparameters for each chunk as a list of tuples
        hyperparams = extract(chunk)
        with Pool(32) as p:
            output = p.starmap(run, hyperparameters)
        print(output)
