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

#	A cada dato se le asigna la clase y un color para diferenciarlas entre sí
for lab, col in zip(mclasses,mcolors):
	plt.scatter(Xred_pca[y==lab, 0],Xred_pca[y==lab, 1],label=lab,c=col)

#	Se configuran las etiquetas
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
leg = plt.legend(loc='upper right', fancybox=True)

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

#	A cada dato se le asigna la clase y un color para diferenciarlas entre sí
for lab, col in zip(mclasses,mcolors):
	plt.scatter(Xred_lda[y==lab, 0],Xred_lda[y==lab, 1],label=lab,c=col)

#	Se configuran las etiquetas
plt.xlabel('LDA/Fisher Direction 1')
plt.ylabel('LDA/Fisher Direction 2')
leg = plt.legend(loc='upper right', fancybox=True)

######## Pregunta (e) ############################################################



######## Pregunta (f) ############################################################



######## Pregunta (g) ############################################################

lda_model = LDA()
lda_model.fit(X_std,y)
print lda_model.score(X_std,y)
print lda_model.score(X_std_test,ytest)
qda_model = QDA()
knn_model = KNeighborsClassifier(n_neighbors=10)

######## Pregunta (h) ############################################################



######## Pregunta (i) ############################################################


#	Se grafican todos los plots anteriores
plt.show()