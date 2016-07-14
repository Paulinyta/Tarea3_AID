import urllib
import pandas as pd
import re, time
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer, word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import random


accuracy_dict_train={}
accuracy_dict_test={}
contador_universal_svm=0
contador_universal_logistic=0
#Pregunta a
print "Pregunta a**********************************************************************************"
train_data_url = "http://www.inf.utfsm.cl/~jnancu/stanford-subset/polarity.train"
test_data_url = "http://www.inf.utfsm.cl/~jnancu/stanford-subset/polarity.dev"
train_data_f = urllib.urlretrieve(train_data_url, "train_data.csv")
test_data_f = urllib.urlretrieve(test_data_url, "test_data.csv")
ftr = open("train_data.csv", "r")
fts = open("test_data.csv", "r")
rows = [line.split(" ",1) for line in ftr.readlines()]
train_df = pd.DataFrame(rows, columns=['Sentiment','Text'])
train_df['Sentiment'] = pd.to_numeric(train_df['Sentiment'])
positivo_train=0
negativo_train=0
for x in train_df['Sentiment']:
	if x==-1:
		negativo_train+=1
	if x==+1:
		positivo_train+=1
print "Sentimiento negativo data entrenamiento: "
print negativo_train
print "Sentimiento positivo date de entrenamiento: "
print positivo_train

rows = [line.split(" ",1) for line in fts.readlines()]
test_df = pd.DataFrame(rows, columns=['Sentiment','Text'])
test_df['Sentiment'] = pd.to_numeric(test_df['Sentiment'])
negativo_test=0
positivo_test=0
for x in test_df['Sentiment']:
	if x==-1:
		negativo_test+=1
	if x==1:
		positivo_test+=1
print "Sentimiento negativo data de test: "
print negativo_test
print "sentimiento positivo data de test: "
print positivo_test
train_df.shape
test_df.shape
print train_df.shape
print test_df.shape






#Stemmer pregunta b
print "Pregunta b**********************************************************************************"
def word_extractor(text):
	commonwords = stopwords.words('english')
	text = re.sub(r'([a-z])\1+', r'\1\1',text)#substitute multiple letter by two
	words = ""
	wordtokens = [ PorterStemmer().stem(word.lower()) \
	for word in word_tokenize(text.decode('utf-8', 'ignore')) ]
	for word in wordtokens:
		if word not in commonwords:
			words+=" "+word
	return words


print "***********Stemmer*************"
print word_extractor("I love to eat cake")
print "I love to eat cake"
print "************************************"
print word_extractor("I love eating cake")
print "I love eating cake"
print "************************************"
print word_extractor("I loved eating the cake")
print "I loved eating the cake"
print "************************************"
print word_extractor("I do not love eating cake")
print "I do not love eating cake"
print "************************************"
print word_extractor("I don't love eating cake")
print "I don't love eating cake"
print "************************************"






#Pregunta c
print "Pregunta c**********************************************************************************"
def word_extractor2(text):
	wordlemmatizer = WordNetLemmatizer()
	commonwords = stopwords.words('english')
	text = re.sub(r'([a-z])\1+', r'\1\1',text)#substitute multiple letter by two
	words = ""
	wordtokens = [ wordlemmatizer.lemmatize(word.lower()) \
	for word in word_tokenize(text.decode('utf-8', 'ignore')) ]
	for word in wordtokens:
		if word not in commonwords:
			words+=" "+word
	return words
print "*********Lematizer**********"
print word_extractor2("I love to eat cake")
print "I love to eat cake"
print "************************************"
print word_extractor2("I love eating cake")
print "I love eating cake"
print "************************************"
print word_extractor2("I loved eating the cake")
print "I loved eating the cake"
print "************************************"
print word_extractor2("I do not love eating cake")
print "I do not love eating cake"
print "************************************"
print word_extractor2("I don't love eating cake")
print "I don't love eating cake"
print "************************************"






#Pregunta d
print "Pregunta d**********************************************************************************"
#Stemmer
texts_train = [word_extractor(text) for text in train_df.Text]
texts_test = [word_extractor(text) for text in test_df.Text]
vectorizer = CountVectorizer(ngram_range=(1, 1), binary='False')
vectorizer.fit(np.asarray(texts_train))
features_train = vectorizer.transform(texts_train)
features_test = vectorizer.transform(texts_test)
labels_train = np.asarray((train_df.Sentiment.astype(float)+1)/2.0)
labels_test = np.asarray((test_df.Sentiment.astype(float)+1)/2.0)
vocab = vectorizer.get_feature_names()
dist=list(np.array(features_train.sum(axis=0)).reshape(-1,))
auxiliar = 0
n_datos = 0
maximo = 0
for tag, count in zip(vocab, dist):
	n_datos+=1
	auxiliar+=count
	if count>maximo:
		maximo=count
promedio=float(auxiliar)/float(n_datos)
utilizadas_train =[]
for tag, count in zip(vocab, dist):
	if count>(promedio+100):
		utilizadas_train.append([tag,count])
print utilizadas_train
auxiliar_test = 0
n_datos_test = 0
maximo_test = 0
dist_test = list(np.array(features_test.sum(axis=0)).reshape(-1,))
for tag, count in zip(vocab, dist_test):
	n_datos_test+=1
	auxiliar_test+=count
	if count>maximo_test:
		maximo_test=count
promedio_test=float(auxiliar_test)/float(n_datos_test)
utilizadas_test =[]
for tag, count in zip(vocab, dist_test):
	if count>(promedio_test+100):
		utilizadas_test.append([tag,count])
print utilizadas_test
#Lematizer
texts_train_lem = [word_extractor2(text) for text in train_df.Text]
texts_test_lem = [word_extractor2(text) for text in test_df.Text]
vectorizer = CountVectorizer(ngram_range=(1, 1), binary='False')
vectorizer.fit(np.asarray(texts_train))
features_train_lem = vectorizer.transform(texts_train_lem)
features_test_lem = vectorizer.transform(texts_test_lem)
labels_train = np.asarray((train_df.Sentiment.astype(float)+1)/2.0)
labels_test = np.asarray((test_df.Sentiment.astype(float)+1)/2.0)
vocab = vectorizer.get_feature_names()
dist=list(np.array(features_train_lem.sum(axis=0)).reshape(-1,))
auxiliar = 0
n_datos = 0
maximo = 0
for tag, count in zip(vocab, dist):
	n_datos+=1
	auxiliar+=count
	if count>maximo:
		maximo=count
promedio=float(auxiliar)/float(n_datos)
utilizadas_train_lem =[]
for tag, count in zip(vocab, dist):
	if count>(promedio+100):
		utilizadas_train_lem.append([tag,count])
print utilizadas_train
auxiliar_test = 0
n_datos_test = 0
maximo_test = 0
dist_test = list(np.array(features_test_lem.sum(axis=0)).reshape(-1,))
for tag, count in zip(vocab, dist_test):
	n_datos_test+=1
	auxiliar_test+=count
	if count>maximo_test:
		maximo_test=count
promedio_test=float(auxiliar_test)/float(n_datos_test)
utilizadas_test_lem =[]
for tag, count in zip(vocab, dist_test):
	if count>(promedio_test+100):
		utilizadas_test_lem.append([tag,count])

print utilizadas_test




#pregunta e
print "Pregunta e**********************************************************************************"
def score_the_model(model,x,y,xt,yt,text):
	global accuracy_dict_test
	global accuracy_dict_train
	global contador_universal_logistic
	global contador_universal_svm
	acc_tr = model.score(x,y)
	acc_test = model.score(xt[:-1],yt[:-1])
	if text!="LOGISTIC" and text!="SVM":
		accuracy_dict_train[text+"_train"]=acc_tr
		accuracy_dict_test[text+"_test"]=acc_test
	if text=="LOGISTIC":
		contador_universal_logistic+=1
		accuracy_dict_train[text+"_train_"+str(contador_universal_logistic)]=acc_tr
		accuracy_dict_test[text+"_test_"+str(contador_universal_logistic)]=acc_test		
	if text=="SVM":
		contador_universal_svm+=1
		accuracy_dict_train[text+"_train_"+str(contador_universal_svm)]=acc_tr
		accuracy_dict_test[text+"_test_"+str(contador_universal_svm)]=acc_test
	print "Training Accuracy %s: %f"%(text,acc_tr)
	print "Test Accuracy %s: %f"%(text,acc_test)
	print "Detailed Analysis Testing Results ..."
	print(classification_report(yt, model.predict(xt), target_names=['+','-']))





#Pregunta f
def do_NAIVE_BAYES(x,y,xt,yt):
	model = BernoulliNB()
	model = model.fit(x, y)
	score_the_model(model,x,y,xt,yt,"BernoulliNB")
	return model
#Stemmer
print "Stemer:"
model=do_NAIVE_BAYES(features_train,labels_train,features_test,labels_test)
test_pred = model.predict_proba(features_test)
spl = random.sample(xrange(len(test_pred)), 15)
for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
	print sentiment, text
#Lematizer
print "---------------------------------------------------------------------"
print "Lematizer"
model_lem=do_NAIVE_BAYES(features_train_lem,labels_train,features_test_lem,labels_test)
test_pred_lem = model_lem.predict_proba(features_test_lem)
spl = random.sample(xrange(len(test_pred_lem)), 15)
for text, sentiment in zip(test_df.Text[spl], test_pred_lem[spl]):
	print sentiment, text


#Pregunta g
print "Pregunta g**********************************************************************************"
def do_MULTINOMIAL(x,y,xt,yt):
	model = MultinomialNB()
	model = model.fit(x, y)
	score_the_model(model,x,y,xt,yt,"MULTINOMIAL")
	return model
#Stemme
print "Stemmer"
model=do_MULTINOMIAL(features_train,labels_train,features_test,labels_test)
test_pred = model.predict_proba(features_test)
spl = random.sample(xrange(len(test_pred)), 15)
for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
	print sentiment, text

#Lematizer
print "---------------------------------------------------------------------"
print "Lematizer"
model_lem=do_MULTINOMIAL(features_train_lem,labels_train,features_test_lem,labels_test)
test_pred_lem = model_lem.predict_proba(features_test_lem)
spl = random.sample(xrange(len(test_pred_lem)), 15)
for text, sentiment in zip(test_df.Text[spl], test_pred_lem[spl]):
	print sentiment, text





#Pregunta h.1
print "Pregunta h.1**********************************************************************************"
def do_LOGIT(x,y,xt,yt):
	start_t = time.time()
	Cs = [0.01,0.1,10,100,1000]
	for C in Cs:
		print "Usando C= %f"%C
		model = LogisticRegression(penalty='l2',C=C)
		model = model.fit(x, y)
		score_the_model(model,x,y,xt,yt,"LOGISTIC")
#Stemmer
print "Stemmer"
do_LOGIT(features_train,labels_train,features_test,labels_test)


#Lematizer
print "---------------------------------------------------------------------"
print "Lematizer"

do_LOGIT(features_train_lem,labels_train,features_test_lem,labels_test)

#Pregunta h.2
print "Pregunta h.2**********************************************************************************"
def do_SVM(x,y,xt,yt):
	Cs = [0.01,0.1,10,100,1000]
	for C in Cs:
		print "El valor de C que se esta probando: %f"%C
		model = LinearSVC(C=C)
		model = model.fit(x, y)
		score_the_model(model,x,y,xt,yt,"SVM")
#Stemmer
print "Stemmer"
do_SVM(features_train,labels_train,features_test,labels_test)

#Lematizer
print "---------------------------------------------------------------------"
print "Lematizer"
do_SVM(features_train_lem,labels_train,features_test_lem,labels_test)


print "__________________________________________________________________________________"
label_test = []
label_train =[]
data_test = []
data_train = []
for x in accuracy_dict_test:
	print "Clasificador | Accuracy_test | Accuracy_train"
	aux=x.split("_",2)
	if aux[0]=='LOGISTIC' or aux[0]=='SVM':
		print "%s |     %f     |     %f"%(x, accuracy_dict_test[x],accuracy_dict_train[aux[0]+"_train_"+aux[2]])
		label_test.append(x)
		label_train.append(aux[0]+aux[2])
		data_train.append(accuracy_dict_train[aux[0]+"_train_"+aux[2]])
		data_test.append(accuracy_dict_test[x])
	if aux[0]=='BernoulliNB' or aux[0]=='MULTINOMIAL' :
		print "%s |     %f     |     %f"%(x, accuracy_dict_test[x],accuracy_dict_train[aux[0]+"_train"])
		label_test.append(x)
		label_train.append(aux[0])
		data_train.append(accuracy_dict_train[aux[0]+"_train"])
		data_test.append(accuracy_dict_test[x])
ayuda=[]
for i in range(1,len(data_test)+1):
	ayuda.append(i)
	print i
plt.plot( ayuda, data_test)
plt.plot( ayuda, data_train)
plt.xticks(ayuda, label_train, rotation=90)
plt.legend(['Train data', 'Test data'], loc='upper left')
plt.gcf().subplots_adjust(bottom=0.25)

plt.show()