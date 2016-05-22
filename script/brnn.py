from keras.models import Model, Graph, Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Merge,TimeDistributedDense,TimeDistributed, merge, GRU, SimpleRNN

import numpy as np
import psiBlast

SS = "HGIEBST "
SS_MAPPING = dict((s,i) for i,s in enumerate(SS))


def getXY(filename, ystr):
	global SS,SS_MAPPING
	x = psiBlast.load_sparse_csr("profile/"+filename+".npz").toarray().T
	y = [[0 for j in range(len(SS))] for i in range(len(ystr))]
	for i,ss in enumerate(ystr):
		y[i][SS_MAPPING[ss]] = 1
	y = np.matrix(y)
	return x,y

def graphModel(input_dim = 28, MAX_SEQ_LENGTH=None,N_CLASSES=8):
	print("FORWARD")
	forward = LSTM(8, input_dim = input_dim,return_sequences=True)
	print("BACKWARD")
	backward = LSTM(8, input_dim = input_dim,go_backwards=True,return_sequences=True)

	print("MODEL")
	model = Graph()
	model.add_input(name='input',input_shape=(MAX_SEQ_LENGTH,input_dim), dtype='float')
	model.add_node(forward, name='forward', input='input')
	model.add_node(backward, name ='backward', input='input')
	model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
	model.add_node(TimeDistributed(Dense(N_CLASSES, activation='softmax')), name='dense', input='dropout')
	model.add_output(name='output', input='dense')

	print("COMPILE")
	model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

	return model

def sequenceModel(input_dim = 28, MAX_SEQ_LENGTH=None,N_CLASSES=8):
	print("FORWARD")
	encoder_a = Sequential()
	encoder_a.add(LSTM(8, input_dim=input_dim,return_sequences=True))
	print("BACKWARD")
	encoder_b = Sequential()
	encoder_b.add(LSTM(8, input_dim=input_dim,go_backwards=True,return_sequences=True))

	print("MODEL")
	model = Sequential()
	model.add(Merge([encoder_a, encoder_b], mode='concat'))
	model.add(TimeDistributed(Dense(N_CLASSES, activation='softmax')))

	print("COMPILE")
	model.compile(loss='categorical_crossentropy',
	            optimizer='rmsprop',
	            metrics=['accuracy'])

	return model

def otherModel(input_dim = 28, MAX_SEQ_LENGTH=None,N_CLASSES=8,Ct=3,NFB = 8, NHO = 11, NHC =9):
	inputs = Input(shape=(None,input_dim))

	print("FORWARD")
	forwardInput = LSTM(output_dim=NHC, input_dim = input_dim,return_sequences=True)(inputs)
	forwardOuput = TimeDistributed(Dense(output_dim=NFB))(forwardInput)

	print("BACKWARD")
	backwardInput = LSTM(output_dim=NHC, input_dim = input_dim,go_backwards=True,return_sequences=True)(inputs)
	backwardOuput = TimeDistributed(Dense(output_dim=NFB))(backwardInput)

	print("CENTER")
	center = TimeDistributed(Dense(output_dim=NHO))(inputs)

	print("MODEL")
	merged = merge([forwardOuput, backwardOuput, center], mode='concat')
	after_dp = Dropout(0.5)(merged)
	output = TimeDistributed(Dense(N_CLASSES, activation='softmax'))(after_dp)

	model = Model(input=inputs, output=output)
	print("COMPILE")
	model.compile(loss='categorical_crossentropy',
	            optimizer='rmsprop',
	            metrics=['accuracy'])

	return model

def main():
	filename = "11ASA"
	ystr = "     HHHHHHHHHHHHHHHHHHHHHH EEE    SEEETTSS S  TTSS    EE  SSSTT  EEE S  TTHHHHHHHHTT  TT EEE EEEEE TT SS  SS  SEEEEEEEEEE  TT  SHHHHHHHHHHHHHHHHHHHHHHHHHS     S SS EEEEHHHHHHHS SS HHHHHHHHHHHHSEEEEE  SSB SSS BSS   TTT   SSB TTSSB SEEEEEEEETTTTEEEEEEEEEE   HHHHHHHHHHHT TTTTTSHHHHHHHTT S  EEEEEEEHHHHHHHHHT S GGGTS     HHHHHHS    "
	x,y = getXY(filename,ystr)



	X_train = np.array([x],ndmin=3)
	Y_train = np.array([y],ndmin=3)


	# model = graphModel()
	# data = {'input':X_train,'output':Y_train}
	# print("FIT")
	# model.fit(data)

	# model = sequenceModel()
	# print("FIT")
	# model.fit([X_train,X_train],Y_train)

	model = otherModel()
	print("FIT")
	hist = model.fit(X_train,Y_train)

if __name__ == '__main__':
	main()




