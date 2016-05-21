import brnn
from random import shuffle
import numpy as np

lr_init = 0.0003 #0.1*2512/827988

def getTrainSet():
	f = open("data/ssTrain50.txt")
	ssTrain = f.readlines()
	f.close()

	X_train,Y_train = [],[]

	for i in range(0,len(ssTrain),3):
		name = ssTrain[i].split()[0].strip(">")
		ss = ssTrain[i+1].strip("\n")
		x,y = brnn.getXY(name,ss)
		X_train.append(x)
		Y_train.append(y)

	return X_train,Y_train

def changeLR(model,lr=None):
	if(lr == None):
		lr = model.optimzer.lr/2
	model.optimizer.lr = lr

def train(model,X_train,Y_train):
	global lr_init

	changeLR(model,lr_init)
	training_order = list(range(len(X_train)))
	nReduction = 0
	loss = None
	epoch = 0
	epochNotImprove = 0

	while(nReduction < 8):
		shuffle(training_order)
		print("Epoch "+str(epoch),end=" : ")
		for i in training_order:
			X = np.array([X_train[i]],ndmin=3)
			Y = np.array([Y_train[i]],ndmin=3)
			hist = model.fit(X,Y,batch_size=1,nb_epoch=1,verbose=0)

		epoch +=1

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
	X_train,Y_train = getTrainSet()
	model = brnn.otherModel()
	train(model,X_train,Y_train)

if __name__ == '__main__':
	main()