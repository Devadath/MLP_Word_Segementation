#!/usr/bin/env python3

import sys
import gensim
import numpy as np

import argparse
from random import shuffle


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from collections import OrderedDict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.layers import TimeDistributed
from sklearn.metrics import accuracy_score
from keras import metrics
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras.layers import Input
from keras import callbacks


class PreProcessing():

	def __init__(self):
		self.char_embeddings = gensim.models.KeyedVectors.load_word2vec_format('mal_char_embeddings_50.txt', binary=False)

	def parse_args(self):
		parser = argparse.ArgumentParser()

		parser.add_argument('--data', default='wiki', type=str)
		parser.add_argument('--percentage', default='100.00', type=float)
		parser.add_argument('--batch_size', default='100', type=int)
		parser.add_argument('--epochs', default='100', type=int)
		args = parser.parse_args()
		print("args are ---", args)
		return args

	def data_from_args(self, args):
		if 'wiki' == args.data:
			train_file = [i.strip() for i in open("./data/27k_only_wiki_pruned_sandhi_char_level_training_data").readlines()]
		elif 'gold' == args.data:
			train_file = [i.strip() for i in open("./data/13.6k_gold_sandhi_char_level_training_data").readlines()]
		elif "full" == args.data:
			train_file = [i.strip() for i in open("./data/40k_gold+wiki_sandhi_char_level_training_data").readlines()]
		else:
			train_file = []

		shuffle(train_file)
		train_data = train_file[:int(args.percentage*len(train_file))]
		return train_data

	def label2vec(self, tags):
		subs = []
		for t in tags:
			if t == "NSP":
				subs.append([0,1])
			else:
				subs.append([1,0])
		return subs

	def contextualiser(self, text, cont):
		main_data = []
		main_labels = []
		just_labels = []
		j = 0
		kkk = 0
		for line in text:
			line, labels = line.strip().split("\t")
			label_list = labels.split()
			just_labels.extend(label_list)
			label_vec = self.label2vec(label_list)
			main_labels.extend(label_vec)
			char_list = line.split()
			if len(char_list) != len(label_list):
				print(char_list, len(char_list), label_list)
			j+=len(char_list)
			char_vec = []
			padding = ['PAD' for i in range(cont)]
			final = padding+char_list+padding
			for ind, char in enumerate(char_list):
				context = final[ind-cont+cont:ind+cont+cont+1]
				embedding_matrix = np.zeros((cont+cont+1, 50))
				for ind, c in enumerate(context):
					if c in self.char_embeddings:
						embedding_matrix[ind] = self.char_embeddings[c]
				main_data.append(np.concatenate(embedding_matrix))
		return np.array(main_data), np.array(main_labels), just_labels


	def pre_process(self, args):
		train_file = self.data_from_args(args)
		print("Collected training file", len(train_file), "samples")
		test_file =  open("./data/char_sandhi_6k_gold_with_indexes").readlines()
		print("Collected test file", len(test_file), "samples")
		valid_file =  open("./data/validation_data").readlines()
		print("Collected validation file", len(valid_file), "samples")
		train_data, train_labels, train_tags = self.contextualiser(train_file, 3)
		print("Processed train data")
		valid_data, valid_labels, valid_tags = self.contextualiser(valid_file, 3)
		print("Processed validation data")
		test_data, test_labels, test_tags = self.contextualiser(test_file, 3)
		print("Processed test data")
		return train_data, train_labels, train_tags, valid_data, valid_labels, valid_tags, test_data, test_labels, test_tags

class MultiLayerPerceptron():

	def __init__(self):
		pass

	def mlp_model(self):
		model = Sequential()
		model.add(Dense(100, activation='relu', input_shape=(350,)))
		model.add(Dropout(0.3))
		model.add(Dense(100, activation='relu'))
		model.add(Dropout(0.3))
		model.add(Dense(2, activation='softmax'))
		return model


	def run_mlp(self, args, train_data, train_labels, valid_data, valid_labels, test_data, test_labels):
		model = self.mlp_model()
		model.summary()
		#callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0)]
		model.compile(loss='categorical_crossentropy',
		              optimizer='adam',
		              metrics=['accuracy'])

		history = model.fit(train_data,train_labels,
		                    batch_size=args.batch_size,
		                    epochs=args.epochs,
		                    verbose=1,
		                    validation_data=(valid_data,valid_labels))
		score = model.evaluate(test_data, test_labels, verbose=0)

		model_json = model.to_json()
		with open("./models/model_mlp_1_100h_2_100h_"+str(args.batch_size)+"batch_"+str(args.epochs)+"epoch_adam_50dim_drop0.3_1_relu_2_relu_3_softmax_on_"+str(args.data)+"_"+str(args.percentage)+"%_3context.json", "w") as json_file:
		    json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights("./models/model_mlp_1_100h_2_100h_"+str(args.batch_size)+"batch_"+str(args.epochs)+"epoch_adam_50dim_drop0.3_1_relu_2_relu_3_softmax_on_"+str(args.data)+"_"+str(args.percentage)+"%_3context.h5")
		print("Saved model to disk")


class Evalauation():
	def __init__(self):
		pass

	def loading_model(self, test_data, test_labels, args):
		json_file = open("./models/model_mlp_1_100h_2_100h_"+str(args.batch_size)+"batch_"+str(args.epochs)+"epoch_adam_50dim_drop0.3_1_relu_2_relu_3_softmax_on_"+str(args.data)+"_"+str(args.percentage)+"%_3context.json", 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		#load weights into new model
		loaded_model.load_weights("./models/model_mlp_1_100h_2_100h_"+str(args.batch_size)+"batch_"+str(args.epochs)+"epoch_adam_50dim_drop0.3_1_relu_2_relu_3_softmax_on_"+str(args.data)+"_"+str(args.percentage)+"%_3context.h5", "r")
		print("Loaded model from disk")

		# evaluate loaded model on test data
		loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		score = loaded_model.evaluate(test_data, test_labels, verbose=0, batch_size=100)
		print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
		return loaded_model, score

	def vec2labels(self, vec):
		word = []
		for c in vec:
			if c[0] > c[1]:	word.append('SP')
			else:	word.append('NSP')
		return word

	def postprocess(self, loaded_model, test_data):
		preds = loaded_model.predict(test_data)
		preds_labels = self.vec2labels(preds.tolist())
		return preds_labels

	def preprocess_target(self, target_list):
		li = preprocessing.LabelEncoder()
		li.fit(['SP', 'NSP'])
		target = li.transform(target_list)	 		
		return target, list(li.classes_)

	def evaluate(self, gold_labels, pred_labels, target_names):
	 	overall_accuracy = accuracy_score(gold_labels, pred_labels)
	 	#class_report = classification_report(gold_labels, pred_labels, target_names=target_names)
	 	precision = metrics.precision_score(gold_labels, pred_labels)
	 	recall = metrics.recall_score(gold_labels, pred_labels)
	 	f1_score = metrics.f1_score(gold_labels, pred_labels)  
	 	return overall_accuracy, precision, recall, f1_score

	def load_and_evaluate(self, test_data, test_labels, test_tags, args):
		print("Evaluating.......")
		model, score = self.loading_model(test_data, test_labels, args)
		preds_labels = self.postprocess(model, test_data)
		gold_labels, a = self.preprocess_target(test_tags)
		pred_labels, b = self.preprocess_target(preds_labels)
		CM = confusion_matrix(gold_labels, pred_labels)
		overall_accuracy, precision, recall, f1_score = self.evaluate(gold_labels, pred_labels, a)
		return score, overall_accuracy, precision, recall, f1_score, CM




if __name__ == '__main__':

	pp = PreProcessing()
	args = pp.parse_args()
	train_data, train_labels, train_tags, valid_data, valid_labels, valid_tags, test_data, test_labels, test_tags = pp.pre_process(args)
	
	mlp = MultiLayerPerceptron()
	mlp.run_mlp(args, train_data, train_labels, valid_data, valid_labels, test_data, test_labels)

	evaluation = Evalauation()
	score, overall_accuracy, precision, recall, f1_score, CM = evaluation.load_and_evaluate(test_data, test_labels, test_tags, args)
	#print(score, overall_accuracy, precision, recall, f1_score)
	TN = CM[0][0]
	FN = CM[1][0]
	TP = CM[1][1]
	FP = CM[0][1]

	outfile = open("./results/model_mlp_1_100h_2_100h_"+str(args.batch_size)+"batch_"+str(args.epochs)+"epoch_adam_50dim_drop0.3_1_relu_2_relu_3_softmax_on_"+str(args.data)+"_"+str(args.percentage)+"%_3context.results", "w")

	outfile.write("Overall Accuracy : "+ str(overall_accuracy*100)+"%"+"\n")
	outfile.write("Precision: "+str(precision)+"\n")
	outfile.write("Recall: "+str(recall)+"\n")
	outfile.write("F1-score: "+str(f1_score)+"\n")
	outfile.write("TN == > "+str(TN)+"\n")
	outfile.write("FN == > "+str(FN)+"\n")
	outfile.write("TP == > "+str(TP)+"\n")
	outfile.write("FP == > "+str(FP))

	print("Overall Accuracy : ", str(overall_accuracy*100)+"%")
	#print("Classification report : ", report)

	print("Precision : ", precision)
	print("Recall : ", recall)
	print("F1-score : ", f1_score)
	print("TN == >", TN)
	print("FN == >", FN)
	print("TP == >", TP)
	print("FP == >", FP)
	print("done")