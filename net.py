#!/usr/bin/python3

import os
import tensorflow
import tflearn
import numpy as np

### disable tensorflow warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tensorflow.logging.set_verbosity(tensorflow.logging.FATAL)

### Net class

class Net ():

	# create a net object
	# takes an array of sizes eg. [20, 2] or [10, 32, 16, 2]
	def __init__(self, sizes, zero=False):

		# reset the network
		self.reset()

		# save the array of sizes
		self.sizes = sizes

		# create model layer by layer - layer 0 is the input layer
		layerCount = len(self.sizes)
		self.layers = []
		for layerNumber, size in enumerate(self.sizes):
			if layerNumber == 0:
				layer = tflearn.input_data(shape=[None, size])
			elif layerNumber == layerCount-1:
				layer = tflearn.fully_connected(previousLayer, size, activation='softmax')
			else:
				layer = tflearn.fully_connected(previousLayer, size, activation='tanh')
			self.layers.append(layer)
			previousLayer = layer
		self.net = tflearn.regression(previousLayer)
		self.model = tflearn.DNN(self.net, tensorboard_verbose=0)

		if zero:
			self.zero()

	# reset network
	def reset(self):

		# reset tensorflow graph
		tensorflow.reset_default_graph()

	########## modifying single weights

	# return a single weight
	def getWeight(self, layerNumber, rowNumber, colNumber):

		return self.getLayerWeights(layerNumber)[rowNumber, colNumber]

	# set a single weight
	def setWeight(self, layerNumber, rowNumber, colNumber, weight):

		weights = self.getLayerWeights(layerNumber)
		weights[rowNumber, colNumber] = weight
		self.setLayerWeights(layerNumber, weights)

	# adjust a single weight
	def adjustWeight(self, layerNumber, rowNumber, colNumber, adjust):

		weights = self.getLayerWeights(layerNumber)
		weights[rowNumber, colNumber] += adjust
		self.setLayerWeights(layerNumber, weights)

	########## modifying all weights

	# set all weights to zero
	def zero(self):
		for layerNumber, layer in enumerate(self.layers):
			if layerNumber != 0:
				weights = np.zeros([self.sizes[layerNumber - 1], self.sizes[layerNumber]])
				self.setLayerWeights(layerNumber, weights)

	# return a single flat list of weights
	def getModelWeights(self):
		result = []
		for layerWeights in self.getAllLayerWeights():
			result += list(layerWeights.flatten())
		return result

	# set weights from a flat list
	def setModelWeights(self, weights):
		position = 0
		previousLayerSize = 0
		for layerNumber, layer in enumerate(self.layers):
			layerSize = self.sizes[layerNumber]
			layerWeightSize = previousLayerSize*layerSize
			if layerNumber != 0:
				layerWeights = weights[position:position+layerWeightSize]
				#print(layerNumber, position, previousLayerSize, layerSize, layerWeightSize, layerWeights)
				self.model.set_weights(layer.W, np.array(layerWeights).reshape([previousLayerSize, layerSize]))
				position += layerWeightSize
			previousLayerSize = layerSize

	# return a numpy array of numpy arrays of layer weights
	def getAllLayerWeights(self):
		result = []
		for layerNumber, layer in enumerate(self.layers):
			result.append(self.getLayerWeights(layerNumber))
		return result

	# set layer weights from a list of numpy arrays
	def setAllLayerWeights(self, weights):
		for layerNumber, layer in enumerate(self.layers):
			if layerNumber != 0:
				self.setLayerWeights(layerNumber, weights[layerNumber])

	########## modifying layers

	# return a numpy array of layer weights 
	def getLayerWeights(self, layerNumber):
		if layerNumber < 1 or layerNumber >= len(self.layers):
			return np.array([])
		return self.model.get_weights(self.layers[layerNumber].W)

	# set layer weights from a numpy array
	def setLayerWeights(self, layerNumber, weights):
		if layerNumber < 1 or layerNumber >= len(self.layers):
			return
		layer = self.layers[layerNumber]
		self.model.set_weights(layer.W, weights.reshape(layer.W.shape))

	# append layer - returns a new net or false
	def appendLayer(self):
		# get the insert point and size
		insertAt = len(self.sizes) - 1
		insertSize = self.sizes[-1]
		# modify sizes
		sizes = self.sizes
		sizes.insert(insertAt, insertSize)
		# create new net of correct size
		newNet = Net(sizes)
		# copy layers up to insert point
		for layerNumber, layer in enumerate(self.layers):
			if layerNumber != 0:
				weights = self.getLayerWeights(layerNumber)
				newNet.setLayerWeights(layerNumber, weights)
		# add identity layer
		weights = np.identity(insertSize)
		newNet.setLayerWeights(insertAt + 1, weights)
		# return new net
		return newNet

	# append node to layer - returns a new net or false
	def appendNodeToLayer(self, layerNumber):
		# cant append to input layer or output layer
		if layerNumber < 1 or layerNumber >= len(self.layers) - 1:
			return None
		# modify sizes
		sizes = self.sizes
		sizes[layerNumber] += 1
		# create new net of correct size
		newNet = Net(sizes)
		# append column to layer
		layerWeights = self.getLayerWeights(layerNumber)
		column = np.zeros((np.shape(layerWeights)[0], 1))
		layerWeights = np.concatenate((layerWeights, column), axis=1)
		newNet.setLayerWeights(layerNumber, layerWeights)
		# append row to next layer
		layerWeights = self.getLayerWeights(layerNumber + 1)
		row = np.zeros((1, np.shape(layerWeights)[1]))
		layerWeights = np.concatenate((layerWeights, row), axis=0)
		newNet.setLayerWeights(layerNumber + 1, layerWeights)
		# return new net
		return newNet

	########## duplication

	# duplicate net - returns a new net
	def duplicate(self):
		# create a new net of the same size
		newNet = Net(self.sizes)
		# copy layers
		for layerNumber, layer in enumerate(self.layers):
			if layerNumber != 0:
				weights = self.getLayerWeights(layerNumber)
				newNet.setLayerWeights(layerNumber, weights)
		# return the duplicate net
		return newNet

	########## print

	# print details of this object
	def print(self):
		print('sizes:', self.sizes)
		for layerNumber, layer in enumerate(self.layers):
			print('layer:', layerNumber, layer)
			print(self.getLayerWeights(layerNumber))
