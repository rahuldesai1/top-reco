import test
import train

class Model():
	def __init__(self, model=None):
		if model is None:
			self.model = "dependencies/model_1"
		else:
			self.model = model

	def train(data_path, output_path, layers=9, batch_size=256, optimizer="tf.train.AdamOptimizer", learning_rate=0.001, dropout=0.5):
		train.train(data_path, output_path, layers, batch_size, optimizer, learning_rate, dropout)

	def classify(data_path, output_path):
		test.run(data_path, output_path)
		
