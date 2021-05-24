import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

def create_dataset():
	# f(x) = x**2+y**2+xy+x+2y-6
	# lets assume: 
	# x1 =  x**2,	x2= y**2,	x3= x,	x4= y,	x5= x*y
	# So f(x) or output = x1+x2+x5+x3+2*x4-6
	output = []
	x1,x2,x3,x4,x5 = [],[],[],[],[]
	for x in range(25):
		for y in range(25):
			temp_x1 =  x**2
			temp_x2= y**2
			temp_x3= x
			temp_x4= y
			temp_x5= x*y
			temp_output = temp_x1+temp_x2+temp_x5+temp_x3+2*temp_x4-6

			x1.append(temp_x1)
			x2.append(temp_x2)
			x3.append(temp_x3)
			x4.append(temp_x4)
			x5.append(temp_x5)
			output.append(temp_output)
	return x1,x2,x3,x4,x5,output


class ANN(nn.Module):
	def __init__(self,input_size,num_classes):
		super(ANN,self).__init__()
		# Inputs to hidden layer linear transformation
		self.hidden = nn.Linear(input_size, 5)
		# Output layer
		self.output = nn.Linear(5, num_classes)
		# Define sigmoid activation and softmax output 
		self.relu = nn.ReLU()
		
	def forward(self, x):
		# Pass the input tensor through each of our operations
		x = self.hidden(x)
		x = self.relu(x)
		x = self.output(x)
		
		return x





# Creating dataset and passing to Dataloader	
x1,x2,x3,x4,x5,output = create_dataset()
input = torch.Tensor(np.column_stack((x1,x2,x3,x4,x5)))
output = torch.Tensor(output)
input_train,input_test,output_train,output_test = train_test_split(input,output,test_size = 0.2, random_state = 4) # Splitting into training and testing dataset
train_dataset = TensorDataset(input_train,output_train) 
train_dataloader = DataLoader(train_dataset)
test_dataset = TensorDataset(input_test,output_test) 
test_dataloader = DataLoader(test_dataset)

# Hyperparameter
input_size = 5
num_classes = 1
learning_rate = 0.01
num_epochs = 1

# intialize model
model = ANN(5,1)
print("model ",model)
print("intial weight ",model.hidden.weight.data)
print("intial bias ",model.hidden.bias.data)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

# Train model

for epoch in range(num_epochs):
	for batch_idx, (data, target) in enumerate(train_dataloader):
		# forward propogation
		score = model(data)
		print("batch_idx, (data, target)",batch_idx, data, target)
		if batch_idx%50 ==0: # print every 50th data
			print("score ",score.data,"target ",target)
		
		loss = criterion(score, target)
		# backward propogation
		optimizer.zero_grad()
		loss.backward()
		#gradient descent or adam step
		optimizer.step()


def check_accuracy(loader,model,type_of_data):
	err_array = []
	num_samples = len(loader)
	model.eval()
	print("Calculating {} data error.....".format(type_of_data))
	with torch.no_grad():
		for x,y in loader:
			prediction = model(x)
			# print("predicted value: ",prediction,"actual value: ",y)
			err_array.append(abs(y-prediction))
	print("Average error: ",sum(err_array)/num_samples)


		
check_accuracy(train_dataloader,model,"training")
check_accuracy(test_dataloader,model,"testing")