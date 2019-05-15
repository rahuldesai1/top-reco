from classifier import Model

#training example where model is saved to default folder in the dependencies
model = Model()

#load the normalized dataset (NOTE: Normalized has been set to True)
data_path = "samples/norm_results.csv"

#Train the model on certain hyper-parameters and don't save the dataset again
#model.train(data_path, save_data=False, normalized=True, delimiter=",", epochs=15, layers=11, batch_size=512)

#testing example using the model that was previously trained
#NOTE: we don't have to re-load the data because the default model is now the most recently trained model
test_data = "samples/norm_results.csv"
model.classify(test_data, normalized=True, delimiter=",", output_path="results/test")

