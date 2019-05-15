import classifier.test
import classifier.train
import classifier.data_processing

class Model():

	def __init__(self, model=None, output_dir=None):
		#model is used for inference
		base_path = "~/projects/classifier/classifier/dependencies/"
		if model is None:
			model = "model_2"
			self.model = base_path + model
			print("[INFO] Using default model: {0}".format(self.model))
		else:
			self.model = model
		#output_dir is used for saving the model to a specific directory after training
		if output_dir is None:
			print("[WARNING] Output directory set to dependencies/default. Training will overwrite any models currently in the default folder. Use set_output_dir(dir) to set the output directory.")
			self.output_dir = base_path + "default"
		else:
			self.output_dir = output_dir

	def train(self, data_path, save_data=False, normalized=False, delimiter=",", epochs=20, layers=9, batch_size=256, optimizer="tf.train.AdamOptimizer", learning_rate=0.001, dropout=0.5):
		if not normalized:
			norm_data = data_processing.process(data_path, self.output_dir, save=save_data, delimiter=delimiter)
		else:
			norm_data = data_processing.load_data(data_path, delimiter)

		model_path = train.train(norm_data, self.output_dir, epochs, layers, batch_size, optimizer, learning_rate, dropout)
		#set the current model to the trained model.
		self.model = model_path
		print("[INFO] Training done. To run inference use the classify function")

	def classify(self, data_path, output_path, normalized=False, delimiter=","):
		if not normalized:
			norm_data = data_processing.normalize(data_path, self.model, delimiter)
		else:
			norm_data = data_processing.load_data(data_path, delimiter)
		test.run(self.model, norm_data, output_path)

	def load_model(self, model):
		self.model = model

	def classify_triplet(self, row, thresh=0.5):
		result = test.classify_triplet(self.model, row, thresh)
		if result == 1:
			print("Classification: Triplet")
		else:
			print("Classification: Background")
		return result
	
	def set_output_dir(self, dir):
		self.output_dir = dir
