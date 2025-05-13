import numpy as np

	
class Network:
	def __init__(self):
		#Prepare the data
		self.elements = np.empty(shape = 5, dtype = object)
		with open('Network_Input.txt', 'r') as file:
			cnt = 0
			for line in file:
				if(cnt == 2):
					nr = 0
					for i in line:
						if i.isdigit():
							nr = nr * 10 + int(i)
					self.elements[2] = np.array([nr])
				else:
					self.elements[cnt] = np.array([int(i) for i in line if i.isnumeric() == True])
				cnt += 1
		
		#Organise the training data
		self.elements[3] = self.elements[3].reshape(self.elements[2][0], int(self.elements[3].size / self.elements[2][0]))
		self.elements[4] = self.elements[4].reshape(self.elements[2][0], 1)
		
		#Prepare the network's structure
		self.layers = np.empty(shape = self.elements[0][0], dtype = object)
		self.weights = np.empty(shape = self.elements[0][0] - 1, dtype = object)
		self.biases = np.empty(shape = self.elements[0][0], dtype = object)
		
		#Initialize layers
		for i in range(self.elements[0][0]):
			self.layers[i] = np.zeros(shape = self.elements[1][i])
			
		#Initialize weights and biases
		for i in range(self.layers.size - 1):
			self.weights[i] = np.random.uniform(-1, 1, (self.layers[i].size, self.layers[i + 1].size))
			self.biases[i + 1] = np.random.uniform(-1, 1, (1, self.layers[i + 1].size))
		
	def activation(self, x, derivative):
		tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
		if(derivative == False):
			return tanh
		else:
			return 1 - tanh * tanh
			
	def foward_prop(self, x):
		self.layers[0] = x
		for i in range(self.layers.size - 1):
			self.layers[i + 1] = self.activation(np.dot(self.layers[i], self.weights[i]) + self.biases[i + 1], False)
			
	def back_prop(self, x, y):
		self.foward_prop(x)
		const = y - self.layers[self.layers.size - 1]
		for i in range(self.layers.size - 2, -1, -1):
			unmodif_w = self.weights[i]
			const = np.multiply(const, self.activation(np.dot(self.layers[i], self.weights[i]) + self.biases[i + 1], True))
			self.weights[i] += 0.1 * np.dot(self.layers[i].reshape(-1, 1), const.reshape(1, -1))
			self.biases[i + 1] += 0.1 * const.reshape(1, -1)
			const = np.dot(const, unmodif_w.T)
			
	def train(self, t_i, t_o):
		percent = 1
		for z in range(int(40000 / self.elements[2][0])):
			if(z % int(4000 / self.elements[2][0]) == 0):
				print(str(z / int(4000 / self.elements[2][0]) * 10) + "%")
			for i in range(self.elements[2][0]):
				self.back_prop(t_i[i], t_o[i])
			
	def get_output(self):
		return self.layers[self.layers.size - 1]
		
network = Network()
network.train(network.elements[3], network.elements[4])

while(True):
	print("Enter input data:")
	input_data = np.zeros(shape = network.elements[1][0])
	for i in range(input_data.size):
		input_data[i] = int(input())
	network.foward_prop(input_data)
	print(network.get_output())
