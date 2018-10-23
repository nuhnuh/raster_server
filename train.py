#!/usr/bin/env python3
"""
"""



import numpy as np
import sklearn
import os
import matplotlib.pyplot as plt
import cv2


# load dataset
fn = 'data.csv'
import pandas as pd
col_names = ['label', 'band1_m', 'band2_m', 'band3_m', 'band4_m']
feat_cols = col_names[1:]
df = pd.read_table( fn, delimiter=',', header=None, names=col_names )
df = sklearn.utils.shuffle( df )
print( 'data:' )
print( df.head() )
print( 'value counts:' )
print( df.label.value_counts() )
#

from sklearn import preprocessing
X = df[ feat_cols ].values
y = df[ col_names[0] ].values
le = preprocessing.LabelEncoder()
le.fit( y )
y = le.transform( y )
print( 'classes:', le.classes_ )
#  lb = preprocessing.LabelBinarizer()
#  lb.fit( y )
#  y = lb.transform( y )
#  print( 'classes:', lb.classes_ )



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2 )



# normalization
scaler = preprocessing.StandardScaler().fit( X_train )
X_train_scaled = scaler.transform( X_train )
X_test_scaled  = scaler.transform( X_test )



# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA( n_components=4 )
lda.fit( X_train_scaled, y_train )



# visualize train set dicriminant components
X_lda = lda.transform( X_train_scaled )
#
x_DC1 = X_lda[:,0]
x_DC2 = X_lda[:,1]
for class_ in le.classes_ :
    I = y_train == le.transform([ class_ ])
    color = np.random.rand(1,3)
    color = tuple( color[0,:] )
    plt.plot( x_DC1[I], x_DC2[I], linestyle='', marker='o', color=color, alpha=.5 )
plt.legend( le.classes_ )
plt.show()

#  #
#  from sklearn.ensemble import RandomForestClassifier
#  classifier = RandomForestClassifier( max_depth=2, random_state=0 )
#  classifier.fit( X_scaled, y )
#  y_pred = classifier.predict( X_scaled )
#  from sklearn.metrics import accuracy_score
#  print('Accuracy {}%'.format( accuracy_score(y, y_pred) ))

#  from sklearn import linear_model
#  lm = linear_model.LinearRegression()
#  model = lm.fit( X_train, y_train )
#  predictions = lm.predict( X_test )



# visualize test set dicriminant components
X_lda = lda.transform( X_test_scaled )
#
x_DC1 = X_lda[:,0]
x_DC2 = X_lda[:,1]
for class_ in le.classes_ :
    I = y_test == le.transform([ class_ ])
    color = np.random.rand(1,3)
    color = tuple( color[0,:] )
    plt.plot( x_DC1[I], x_DC2[I], linestyle='', marker='o', color=color, alpha=.5 )
plt.legend( le.classes_ )
plt.show()



#  from sklearn.svm import SVC
#  clf = SVC( gamma='scale', decision_function_shape='ovo' )
#  clf.fit( X_train_scaled, y_train ) 
#  #
#  y_train_pred = clf.predict( X_train_scaled )
#  y_test_pred  = clf.predict( X_test_scaled )
#  #
#  from sklearn.metrics import accuracy_score
#  print('Train accuracy {}%'.format( accuracy_score(y_train, y_train_pred) ))
#  print('Test accuracy {}%'.format( accuracy_score(y_test, y_test_pred) ))



from sklearn.svm import SVC
clf = SVC( gamma='scale', decision_function_shape='ovo' )
X_train_lda = lda.transform( X_train_scaled )
X_test_lda  = lda.transform( X_test_scaled )
clf.fit( X_train_lda, y_train ) 
#
y_train_pred = clf.predict( X_train_lda )
y_test_pred  = clf.predict( X_test_lda )
#
from sklearn.metrics import accuracy_score
print('Train accuracy {}%'.format( accuracy_score(y_train, y_train_pred) ))
print('Test accuracy {}%'.format( accuracy_score(y_test, y_test_pred) ))




#
import sys
sys.exit(0)











import seaborn as sns
sns.pairplot( df, hue='label' )
plt.show()




#  from sklearn.decomposition import FactorAnalysis
#  feat_cols = col_names[1:]
#  FA = FactorAnalysis(n_components = 2).fit_transform( df[feat_cols].values )



from sklearn.decomposition import PCA
pca = PCA( n_components=len(feat_cols) )
pca_result = pca.fit_transform( X_scaled )
plt.plot( range(4), pca.explained_variance_ratio_ )
plt.plot( range(4), np.cumsum(pca.explained_variance_ratio_) )
plt.title( 'Component-wise and Cumulative Explained Variance' )
plt.show()



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA( n_components=2 )
lda.fit( X_scaled, y )
X_lda = lda.transform( X_scaled )
#
df['y'] = y
df['PC1'] = X_lda[:,0]
df['PC2'] = X_lda[:,1]
#  sns.regplot( data=df[['PC1','label']], x = 'PC1',y = 'label', fit_reg=False, scatter_kws={'s':50} )
sns.regplot( data=df[['PC1','y']], x='PC1', y='y' )
plt.show()


for class_ in le.classes_ :
    I = df['label'].values == class_
    plt.plot( df['PC1'].values[I], df['PC2'].values[I], color=np.random.rand(1,3) )
plt.show()




#
#  for color, i, target_name in zip(colors, [0, 1, 2], target_names):
#      plt.hist( X_lda[y == i, 0], color=color, alpha=.5,
#                  label=target_name ) # check lda.decision_function(X_[y_ == i])
#  plt.legend(loc='best')
#  plt.xlabel('DC 1') # discriminat component (DC)
#  plt.title('LDA')
#  plt.show()



#  def _main() :
#      pass
#  
#  if __name__ == '__main__':
#      _main()
