Top Classification with Deep Learning

The following package includes the necessary commands for train and testing a model for top reconstruction. This model is based on a neural network built in Tensorflow. 

Installing Dependencies
1) git clone
2) cd classifier
3) pip install -r requirements.txt
4) python3 setup.py install

Running the Classifier
1) from classifier import Classifier
2) If you would like to use the default model:
	model = Classifier()
3) If you would like to use your own model:
	model = Classifier(path-to-model)
4) model.run(data-path, output-path) #see Section: "Input Data Format" to see
	how input data should be formatted
5) The results should be saved in a .txt file in out-path


Training the Classifier
1) Follow steps 1-3 from above. 
2) model.train(data-path, output-path, ...)
3) Optional Parameters:
	Number of Layers: layers (default=9)
	Batch Size: batch_size (default=256)
	Optimizer: optimizer (default=tf.train.AdamOptimizer) #should be in the form tf.train...
	Learning Rate: learning_rate (default=0.001)
	Dropout: dropout (default=0.5) #keep probability
4) The results of training at each epoch and the points for the ROC curve should be
saved to the output-path in a .txt file. 
5) The weights of the model will also be saved in the output-path. To run this model 
follow the instructions outlined above. 

Input Data Format
1) Input data must be a csv that uses "," as a delimiter
2) The csv file must include the following features in this order:
	W_B_pt, W_B_pt, W_B_eta, W_B_phi, W_B_E, W_B_m, b_B_pt, b_B_eta, b_B_phi, b_B_E
	b_B_m, Wjj_B_dR, tWb_B_dR, Wjj_B_deta, Wjj_B_dphi, tWb_B_deta, tWb_B_dphi, W_pt
	W_eta, W_phi, W_E, W_m, b_pt, b_eta, b_phi, b_E, b_m, Wjj_dR, tWb_dR, Wjj_deta
	Wjj_dphi, tWb_deta, tWb_dphi, t_pt, t_eta, t_phi, t_E, t_m, btag1, btag2, btag3
