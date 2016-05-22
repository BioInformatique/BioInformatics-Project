import theano
theano.config.floatX = 'float32'
theano.config.mode= 'FAST_RUN'
import brnn
from random import shuffle
import numpy as np
import sys
import math
import time as time


lr_init = 0.00015 #0.1*2512/827988

def getTrainSet():
	f = open("data/ssTrain50.txt")
	ssTrain = f.readlines()
	f.close()

	X_train,Y_train = [],[]

	for i in range(0,len(ssTrain),3):
		name = ssTrain[i].split()[0].strip(">")
		ss = ssTrain[i+1].strip("\n")
		x,y = brnn.getXY(name,ss)
		if(x.shape[0]==y.shape[0] and x.shape[0]<300):
			X_train.append(x)
			Y_train.append(y)

	return X_train,Y_train

def changeLR(model,lr=None):
	if(lr == None):
		lr = model.optimzer.lr/2
	model.optimizer.lr = lr
	model.optimizer.epsilon = 1.0e-7

def blocks(X_train,Y_train):
	order = list(range(len(X_train)))
	shuffle(order)

	X_blocks = []
	Y_blocks = []
	i=0
	tmp = []
	maxLen = 0
	nAA = 0
	while i < len(X_train):
		size = X_train[order[i]].shape[0]
		if(maxLen<size):
			maxLen = size
		nAA += size
		tmp += [order[i]]

		if(nAA > 827988 or i == len(X_train)-1):
			X_block = []
			Y_block = []
			for j in tmp:
				shapeX = X_train[j].shape
				shapeY = Y_train[j].shape
				X_block.append(np.concatenate((X_train[j],np.zeros((maxLen-shapeX[0],shapeX[1]),dtype=float))))
				Y_block.append(np.concatenate((Y_train[j],np.zeros((maxLen-shapeY[0],shapeY[1]),dtype=float))))
			X_blocks.append(np.array(X_block))
			Y_blocks.append(np.array(Y_block))
			nAA=0
			tmp=[]
			maxLen =0
		i+=1
	return X_blocks,Y_blocks

def blocks2(X_train,Y_train):
	order = list(range(len(X_train)))
	shuffle(order)

	X_blocks = []
	Y_blocks = []
	for k in range(300):
		i=0
		tmp = []
		maxLen = 0
		nAA = 0
		while i < len(X_train):
			size = X_train[order[i]].shape[0]
			if size == k:
				if(maxLen<size):
					maxLen = size
				nAA += size
				tmp += [order[i]]

			if(nAA > 827988 or i == len(X_train)-1) and tmp:
				X_block = []
				Y_block = []
				for j in tmp:
					shapeX = X_train[j].shape
					shapeY = Y_train[j].shape
					X_block.append(np.concatenate((X_train[j],np.zeros((maxLen-shapeX[0],shapeX[1]),dtype=float))))
					Y_block.append(np.concatenate((Y_train[j],np.zeros((maxLen-shapeY[0],shapeY[1]),dtype=float))))
				X_blocks.append(np.array(X_block))
				Y_blocks.append(np.array(Y_block))
				nAA=0
				tmp=[]
				maxLen =0
			i+=1
	return X_blocks,Y_blocks

def printProgresse(ratio,i,loss,idy,t1,t0, acc):
	sys.stdout.write('\r')
	sys.stdout.write("[%-100s] %d%% loss : %.4f - acc %.4f : - time %d - id : %4d" % ('='*int(i*ratio), int(i*ratio),loss,acc,t1-t0,idy))
	sys.stdout.flush()

def train(model,X_train,Y_train):
	global lr_init
	RATIO = 100.0/len(X_train)
	changeLR(model,lr_init)
	training_order = list(range(len(X_train)))
	# shuffle(training_order)

	nReduction = 0
	loss = None
	epoch = 0
	epochNotImprove = 0

	while(nReduction < 8):
		shuffle(training_order)
		print("Epoch "+str(epoch))
		t0 = time.time()
		for times,i in enumerate(training_order):
			if(len(X_train[i].shape) < 3):
				X = np.array([X_train[i]],ndmin=3)
			else:
				X = X_train[i]
			if(len(Y_train[i].shape) < 3):
				Y = np.array([Y_train[i]],ndmin=3)
			else:
				Y = Y_train[i]
			if(len(X.shape) != 3 or len(Y.shape) != 3):
				print("ERROR")
				continue
			hist = model.fit(X,Y,batch_size=3,nb_epoch=1,verbose=0)

			if(math.isnan(hist.history["loss"][-1])):
				print()
				print("ERROR")
				print(i)
				break

			# data = {'input':X,'output':Y}
			# hist = model.fit(data,batch_size=3,nb_epoch=1,verbose=1)

			# hist = model.fit([X,X],Y,batch_size=3,nb_epoch=1,verbose=1)

			printProgresse(RATIO,times,hist.history["loss"][-1],i,time.time(),t0,hist.history["acc"][-1])
		print()
		epoch +=1

		if(math.isnan(hist.history["loss"][-1])):
				break

		last_loss = hist.history["loss"][-1]
		print(last_loss)
		if(loss == None or last_loss < loss):
			loss = last_loss
			epochNotImprove = 0
			model.save_weights('model/best.h5',overwrite=True)
		else:
			epochNotImprove += 1

		if(epochNotImprove >= 50):
			model.load_weights("model/best.h5")
			changeLR(model)
			epochNotImprove = 0
			epoch -= 50
			nReduction += 1
			print("\n REDUCTION \n")

def main():
	blacklist = [5,101,648,913,1022,1214]
	blacklist.reverse()
	X_train,Y_train = getTrainSet()
	for i in blacklist:
		del X_train[i]
		del Y_train[i]

	# X_blocks,Y_blocks = blocks(X_train,Y_train)
	X_blocks,Y_blocks = blocks2(X_train,Y_train)

	print(len(X_blocks))
	print(X_blocks[0].shape)
	print(Y_blocks[0].shape)
	model = brnn.otherModel()
	# model = brnn.graphModel()
	# model = brnn.sequenceModel()

	train(model,X_blocks,Y_blocks)
	# train(model,X_train,Y_train)

if __name__ == '__main__':
	main()