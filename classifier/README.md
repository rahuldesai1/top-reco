Top Classification with Deep Learning

The following package includes the necessary commands for train and testing a model for top reconstruction. This model is based on a neural network built in Tensorflow. For an example on how to use the package, see sample.py

Installing Dependencies
1) git clone
2) cd classifier
3) pip install -r requirements.txt
4) python3 setup.py install

Constructor: 
1) When defining the model you may specify two things; however, they are optional. If not specified, the model will use the default locations. 
    a) model: path to the directory that stores the graph and the model weights for testing the model
    b) Output_dir: path to directory for saving the model and results after TRAINING
    Note: the default path is dependences/default. If there already exists a model trained in this path, trianing again will overwrite this. 
    
Running the Classifier
1) from classifier import Classifier
2) If you would like to use the default model:
	model = Model()
3) If you would like to use your own model:
	model = Model(path-to-model)
4) model.run(data-path, output-path) #see Section: "Input Data Format" to see
	how input data should be formatted
        a) The output-path determines the directory where the output of the classifier will be saved for different threshold values. 
5) The results should be saved in a .json file in out-path


Training the Classifier
1) from classifier import Classifier
2) If you would like to use the default model:
    model = Model() #see "constructor" above for more details
2) model.train(data-path, output-path, ...)
	Data-path: Path to csv file containing the data
	Output-path: Directory that you want to save the model and results in. 
3) Optional Parameters:
	Number of Epochs: epochs (default=20) #number of epoch that you want to train the network for
	Number of Layers: layers (default=9) #depth of the network
	Batch Size: batch_size (default=256)
	Optimizer: optimizer (default=tf.train.AdamOptimizer) #should be in the form tf.train...
	Learning Rate: learning_rate (default=0.001)
	Dropout: dropout (default=0.5) #keep probability
4) The results of training at each epoch and the points for the ROC curve should be
saved to the output-path in a .txt file. 
5) The weights of the model will also be saved in the output-path. To run this model 
follow the instructions outlined above. 

Input Data Format for testing on pre-trained model
1) Input data must be a csv. 
    NOTE: If the data has already been normalized, please set normalized=True and save_data=False so that the data is not saved to the output path
2) The csv file must include the following features in this order for the default classifier:
	W_B_pt, W_B_pt, W_B_eta, W_B_phi, W_B_E, W_B_m, b_B_pt, b_B_eta, b_B_phi, b_B_E
	b_B_m, Wjj_B_dR, tWb_B_dR, Wjj_B_deta, Wjj_B_dphi, tWb_B_deta, tWb_B_dphi, W_pt
	W_eta, W_phi, W_E, W_m, b_pt, b_eta, b_phi, b_E, b_m, Wjj_dR, tWb_dR, Wjj_deta
	Wjj_dphi, tWb_deta, tWb_dphi, t_pt, t_eta, t_phi, t_E, t_m, btag1, btag2, btag3
3) Make sure that the format of the data that is used for training is the same as the format of the data used for inference.

Output
The output of training will be formatted as such in the output directory: 
1) The necessary files for storing the graph and weights of the model for test time
2) model_results.txt contains the probability distribution of all the points in case you ever need to plot it
3) model_metrics.json contains the points need to draw the ROC (signal efficiency and background acceptance at 100 threshold values)
	You can plot these values using the Data jupyter notebook
 
