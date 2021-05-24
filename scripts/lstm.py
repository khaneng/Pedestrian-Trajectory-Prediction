import numpy 
import numpy as np
#import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sgan.data.trajectories import TrajectoryDataset,read_file,poly_fit
from sgan.utils import relative_to_abs, get_dset_path
import argparse
import os
parser = argparse.ArgumentParser()
dataset_name='zara1'
train_path = get_dset_path(dataset_name, 'train')
test_path = get_dset_path(dataset_name, 'test')

def ped_seq(path):
	skip = 1
	threshold=0.002
	pred_len = 12
	all_files = os.listdir(path)
	all_files = [os.path.join(path, _path) for _path in all_files]
	delim = 'tab'
	for path in all_files:
		data = read_file(path, delim)
		frames = np.unique(data[:, 0]).tolist()
		frame_data = []
		print("frames",len(frames))
		for frame in frames:
			frame_data.append(data[frame == data[:, 0], :])
		num_sequences = int(
			math.ceil((len(frames) - 20 + 1) / skip))
		print("num_sequences",num_sequences)
		for idx in range(0, num_sequences  + 1, skip):
			curr_seq_data = np.concatenate(
				frame_data[idx:idx + 20], axis=0)
			peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
			curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
									 20))
			curr_seq = np.zeros((len(peds_in_curr_seq), 2, 20))
			curr_loss_mask = np.zeros((len(peds_in_curr_seq),
									   20))
			num_peds_considered = 0
			_non_linear_ped = []
			for _, ped_id in enumerate(peds_in_curr_seq):
				curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
											 ped_id, :]
				curr_ped_seq = np.around(curr_ped_seq, decimals=4)
				pad_front = frames.index(curr_ped_seq[0, 0]) - idx
				pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
				if pad_end - pad_front != 20:
					continue
				curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
				curr_ped_seq = curr_ped_seq
				# Make coordinates relative
				rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
				rel_curr_ped_seq[:, 1:] = \
					curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
				_idx = num_peds_considered
				curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
				curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
				# Linear vs Non-Linear Trajectory
				_non_linear_ped.append(
					poly_fit(curr_ped_seq, pred_len, threshold))
				curr_loss_mask[_idx, pad_front:pad_end] = 1
				num_peds_considered += 1
				print("curr_ped_seq",curr_ped_seq,"ped_id",ped_id)
				return curr_ped_seq


# load the dataset
dataframe = pandas.read_csv('F:\\MTECH\\Thesis\\sgan\\scripts\\airpassenger.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

dset = ped_seq(train_path)
train = [l.tolist() for l in dset[0]]
train = np.reshape(train, (-1, 1))
tset = ped_seq(test_path)
test = [l.tolist() for l in tset[0]]
test = np.reshape(test, (-1, 1))
print("dset",train)


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# print(train, len(test))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print("trainX",trainX,"trainY",trainY)


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))



# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()