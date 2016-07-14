import urllib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import zero_one_loss

######## Pregunta (a) ############################################################

#	Se cargan los datos desde un URL
train_data_url = "http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/vowel.train"
test_data_url = "http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/vowel.test"

#	Los datos se pasan a archivos csv
train_data_f = urllib.urlretrieve(train_data_url, "train_data.csv")
test_data_f = urllib.urlretrieve(test_data_url, "test_data.csv")

#	Los archivos csv se guardan en dataframes para train y test segun corresponda
train_df = pd.DataFrame.from_csv('train_data.csv',header=0,index_col=0)
test_df = pd.DataFrame.from_csv('test_data.csv',header=0,index_col=0)

#	Se imprime la dimension de cada dataframe, mostrando (num de filas, num columnas)
#print train_df.shape
#print test_df.shape

#print train_df
#print train_df.tail()
#print train_df.describe()

######## Pregunta (b) ############################################################

#	Se construyen matrices para los datos de entrenamiento y de test
X = train_df.ix[:, 'x.1':'x.10'].values
y = train_df.ix[:,'y'].values

Xtest = test_df.ix[:,'x.1':'x.10'].values
ytest = test_df.ix[:,'y'].values

#	Se normalizan los datos de entrenamiento y de test
X_std = StandardScaler().fit_transform(X)
X_std_test = StandardScaler().fit_transform(Xtest)

######## Pregunta (c) ############################################################

#	Se utiliza PCA con dos componentes principales
sklearn_pca = PCA(n_components=2)

#	Se ajusta a los datos de entrenamiento
Xred_pca = sklearn_pca.fit_transform(X_std)

#	Se escoge la paleta de colores
cmap = plt.cm.get_cmap('hsv')

#	Se definen las clases
mclasses=(1,2,3,4,5,6,7,8,9)
mcolors = [cmap(i) for i in np.linspace(0,1,10)]

#	Se establece el tamanno de la figura
plt.figure(figsize=(12, 8))

#	A cada dato se le asigna la clase y un color para diferenciarlas entre si
for lab, col in zip(mclasses,mcolors):
	plt.scatter(Xred_pca[y==lab, 0],Xred_pca[y==lab, 1],label=lab,c=col)

#	Se configuran las etiquetas
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
leg = plt.legend(loc='upper right', fancybox=True)

#yhat_pca = sklearn_pca.predict(X_std_test)
#print accuracy_score(ytest, yhat_pca)

######## Pregunta (d) ############################################################

#	Se utiliza LDA con dos dimensiones
sklearn_lda = LDA(n_components=2)

#	Se ajusta a los datos de entrenamiento
Xred_lda = sklearn_lda.fit_transform(X_std,y)

#	Se escoge la paleta de colores
cmap = plt.cm.get_cmap('hsv')

#	Se definen las clases
mclasses=(1,2,3,4,5,6,7,8,9)
mcolors = [cmap(i) for i in np.linspace(0,1,10)]

#	Se establece el tamanno de la figura
plt.figure(figsize=(12, 8))

#	A cada dato se le asigna la clase y un color para diferenciarlas entre si
for lab, col in zip(mclasses,mcolors):
	plt.scatter(Xred_lda[y==lab, 0],Xred_lda[y==lab, 1],label=lab,c=col)

#	Se configuran las etiquetas
plt.xlabel('LDA/Fisher Direction 1')
plt.ylabel('LDA/Fisher Direction 2')
leg = plt.legend(loc='upper right', fancybox=True)

######## Pregunta (e) ############################################################

# Conceptual

######## Pregunta (f) ############################################################

probas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for index, row in train_df.iterrows():
	if row['y']==1:
		probas[0]=probas[0]+1
	if row['y']==2:
	  	probas[1]=probas[1]+1
	if row['y']==3:  
		probas[2]=probas[2]+1
	if row['y']==4:  
		probas[3]=probas[3]+1
	if row['y']==5: 
		probas[4]=probas[4]+1
	if row['y']==6:  
		probas[5]=probas[5]+1
	if row['y']==7:  
		probas[6]=probas[6]+1
	if row['y']==8:  
		probas[7]=probas[7]+1
	if row['y']==9:
		probas[8]=probas[8]+1

for i in range(0,9): 
	probas[i]=probas[i]/528

yhat_apriori = np.argmax(probas) + 1

print "Clase: %d"%yhat_apriori

######## Pregunta (g) ############################################################

lda_model = LDA()
lda_model.fit(X_std,y)
print "Score LDA train: %f"%lda_model.score(X_std,y)
print "Score LDA test: %f"%lda_model.score(X_std_test,ytest)
qda_model = QDA()
qda_model.fit(X_std,y)
print "Score QDA train: %f"%qda_model.score(X_std,y)
print "Score QDA test: %f"%qda_model.score(X_std_test,ytest)
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_std,y)
print "Score KNN train: %f"%knn_model.score(X_std,y)
print "Score KNN test: %f"%knn_model.score(X_std_test,ytest)

values_train = []
values_test = []
for i in range(1, 12):
	knn_model = KNeighborsClassifier(n_neighbors=i)
	knn_model.fit(X_std,y)
	values_train.append(knn_model.score(X_std,y))

for i in range(1, 12):
	knn_model = KNeighborsClassifier(n_neighbors=i)
	knn_model.fit(X_std,y)
	values_test.append(knn_model.score(X_std_test,ytest))

plt.figure(figsize=(12, 8))
plt.plot(values_train, label="Training set")
plt.plot(values_test, label="Test set")

plt.xlabel('k')
plt.ylabel('Score')
leg = plt.legend(loc='upper right', fancybox=True)

######## Pregunta (h) ############################################################

lda_train = []
lda_test = []
qda_train = []
qda_test = []
knn_train = []
knn_test = []

lda_model = LDA()
qda_model = QDA()
knn_model = KNeighborsClassifier(n_neighbors=7)
for i in range(1,11):
	sklearn_pca = PCA(n_components=i)
	Xred_pca = sklearn_pca.fit_transform(X_std)
	Xred_pca_test = sklearn_pca.fit_transform(X_std_test)

	lda_model.fit(Xred_pca,y)
	qda_model.fit(Xred_pca,y)
	knn_model.fit(Xred_pca,y)

	yhat_train = lda_model.predict(Xred_pca)
	lda_train.append(zero_one_loss(y, yhat_train)) 
	yhat_test = lda_model.predict(Xred_pca_test)
	lda_test.append(zero_one_loss(ytest, yhat_test)) 

	yhat_train = qda_model.predict(Xred_pca)
	qda_train.append(zero_one_loss(y, yhat_train)) 
	yhat_test = qda_model.predict(Xred_pca_test)
	qda_test.append(zero_one_loss(ytest, yhat_test)) 

	yhat_train = knn_model.predict(Xred_pca)
	knn_train.append(zero_one_loss(y, yhat_train)) 
	yhat_test = knn_model.predict(Xred_pca_test)
	knn_test.append(zero_one_loss(ytest, yhat_test)) 

plt.figure(figsize=(12, 8))
plt.plot(lda_train, label="Training set")
plt.plot(lda_test, label="Test set")

plt.xlabel('d')
plt.ylabel('Error')
leg = plt.legend(loc='upper right', fancybox=True)

plt.figure(figsize=(12, 8))
plt.plot(qda_train, label="Training set")
plt.plot(qda_test, label="Test set")

plt.xlabel('d')
plt.ylabel('Error')
leg = plt.legend(loc='upper right', fancybox=True)

plt.figure(figsize=(12, 8))
plt.plot(knn_train, label="Training set")
plt.plot(knn_test, label="Test set")

plt.xlabel('d')
plt.ylabel('Error')
leg = plt.legend(loc='upper right', fancybox=True)

######## Pregunta (i) ############################################################

lda_train = []
lda_test = []
qda_train = []
qda_test = []
knn_train = []
knn_test = []

lda_model = LDA()
qda_model = QDA()
knn_model = KNeighborsClassifier(n_neighbors=7)
for i in range(1,11):
	sklearn_lda = LDA(n_components=i)
	Xred_pca = sklearn_lda.fit_transform(X_std, y)
	Xred_pca_test = sklearn_lda.fit_transform(X_std_test, ytest)
	lda_model.fit(Xred_pca,y)
	qda_model.fit(Xred_pca,y)
	knn_model.fit(Xred_pca,y)

	yhat_train = lda_model.predict(Xred_pca)
	lda_train.append(zero_one_loss(y, yhat_train)) 
	yhat_test = lda_model.predict(Xred_pca_test)
	lda_test.append(zero_one_loss(ytest, yhat_test)) 

	yhat_train = qda_model.predict(Xred_pca)
	qda_train.append(zero_one_loss(y, yhat_train)) 
	yhat_test = qda_model.predict(Xred_pca_test)
	qda_test.append(zero_one_loss(ytest, yhat_test)) 

	yhat_train = knn_model.predict(Xred_pca)
	knn_train.append(zero_one_loss(y, yhat_train)) 
	yhat_test = knn_model.predict(Xred_pca_test)
	knn_test.append(zero_one_loss(ytest, yhat_test)) 

plt.figure(figsize=(12, 8))
plt.plot(lda_train, label="Training set")
plt.plot(lda_test, label="Test set")

plt.xlabel('d')
plt.ylabel('Error')
leg = plt.legend(loc='upper right', fancybox=True)

plt.figure(figsize=(12, 8))
plt.plot(qda_train, label="Training set")
plt.plot(qda_test, label="Test set")

plt.xlabel('d')
plt.ylabel('Error')
leg = plt.legend(loc='upper right', fancybox=True)

plt.figure(figsize=(12, 8))
plt.plot(knn_train, label="Training set")
plt.plot(knn_test, label="Test set")

plt.xlabel('d')
plt.ylabel('Error')
leg = plt.legend(loc='upper right', fancybox=True)

#	Se grafican todos los plots anteriores
plt.show()