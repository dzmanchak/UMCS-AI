import numpy as np 

class NeuralNetwork:
	def __init__(self):
		self.w0=np.random.normal()
		self.w1=np.random.normal()
		self.w2=np.random.normal()
		self.w3=np.random.normal()
		self.w4=np.random.normal()
		self.w5=np.random.normal()
		self.b1=np.random.normal()
		self.b2=np.random.normal()
		self.b3=np.random.normal()

	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def derivsigmoid(self, x):
		return self.sigmoid(x) * (1 - self.sigmoid(x))

	def mse_loss(self, y, y_pred):
		return np.mean((y - y_pred) ** 2)

	def feedforward1(self,x):
		H1 = self.sigmoid(x[0] * self.w0 + x[1] * self.w1 + self.b1)
		H2 = self.sigmoid(x[0] * self.w2 + x[1] * self.w3 + self.b2)
		return self.sigmoid(H1 * self.w4 + H2 * self.w5 + self.b3)

	def feedforward(self,x1 , x2 , b, w1, w2):
		return (x1 * w1 + x2 * w2 + b)
		
	def MSE(self ,y ,y_pred):
		return -2 * (y - y_pred)

	def train(self, start_data, end_data, learn_rate, epochs, data_count):
		for epoch in range(epochs):
			for i in range(data_count):
				x = np.array([start_data[i][0] ,start_data[i][1] ])
				fH1 = self.feedforward(x[0],x[1],self.b1, self.w0, self.w1)
				H1 = self.sigmoid(fH1)
				fH2 = self.feedforward(x[0],x[1],self.b2, self.w2, self.w3)
				H2 = self.sigmoid(fH2)
				fO1 = self.feedforward(H1, H2 , self.b3 , self.w4 , self.w5)
				O1 = self.sigmoid(fO1)
				y_pred = O1
				MSE = self.MSE(end_data[i] , y_pred) #Dobrze

				dW5 = H2 *  self.derivsigmoid(fO1)
				dW4 = H1 *  self.derivsigmoid(fO1)
				db3 = self.derivsigmoid(fO1)

				dH1 = self.w4 * self.derivsigmoid(fO1)
				dH2 = self.w5 * self.derivsigmoid(fO1)

				dW0 = x[0] * self.derivsigmoid(fH1)
				dW1 = x[1] * self.derivsigmoid(fH1)
				db1 =self.derivsigmoid(fH1)

				dW2 = x[0] * self.derivsigmoid(fH2)
				dW3 = x[1] * self.derivsigmoid(fH2)
				db2 = self.derivsigmoid(fH2)


				self.w0 = self.w0 - learn_rate * dW0 * MSE * dH1
				self.w1 = self.w1 - learn_rate * dW1 * MSE * dH1
				self.w2 = self.w2 - learn_rate * dW2 * MSE * dH2
				self.w3 = self.w3 - learn_rate * dW3 * MSE * dH2
				self.w4 = self.w4 - learn_rate * dW4 * MSE
				self.w5 = self.w5 - learn_rate * dW5 * MSE
				self.b1 = self.b1 - learn_rate * db1 * MSE * dH1
				self.b2 = self.b2 - learn_rate * db2 * MSE * dH2
				self.b3 = self.b3 - learn_rate * db3 * MSE

			if epoch % 10 == 0:
				YPRED = np.array([self.feedforward1(start_data[0]),self.feedforward1(start_data[1]),self.feedforward1(start_data[2]),self.feedforward1(start_data[3])])
				MSE_loss = self.mse_loss(end_data , YPRED)
				print("Epoch : " + str(epoch) + " MSE : " + str(MSE_loss))
				print(YPRED)

start_data = np.array([[0,0],
	[0,1],
	[1,0],
	[1,1]])
end_data = np.array([0 , 1 , 1, 1])
epochs = 10000
learn_rate = 0.1
data_count = 4
network = NeuralNetwork()
network.train(start_data , end_data, learn_rate , epochs , data_count)