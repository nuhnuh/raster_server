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





# TODO
def load_dataset() :
    X_files, y = ds
    X = []
    for fn in X_files :
        img = cv2.imread( fn )[..., ::-1]
        img = np.float32( img )
#         img = img[...,0]/255 + img[...,1]/255 + img[...,2]/255\n",
        X.append( img )
    X = np.stack( X, axis=0 )
    return X, y
def create_model( input_shape ) :

    from keras.models import Model
    from keras.layers import Cropping2D, Lambda, Reshape
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Input, Dense, ELU, Activation, Flatten
    from keras.layers import Dropout, BatchNormalization

    from keras import backend as K
    print('input_shape', input_shape)
    input_ = Input( shape=input_shape )
#     hidden = Lambda( lambda x: x[:,:,:,0]/3 + x[:,:,:,1]/3 + x[:,:,:,2]/3 )( input_ )\n",
#     hidden = Reshape( (*input_shape[:-1], 1) )( hidden )\n",
    hidden = input_
    hidden = BatchNormalization()( hidden )
#     hidden = Cropping2D( cropping=((0,0),(0,0)) )( input_ )
    with K.name_scope('Layer1') :
        hidden = Conv2D( 16, 3 )( hidden )
        print( 'Layer 1:', hidden.shape )
        hidden = MaxPooling2D( pool_size=(2, 2) )( hidden )
        hidden = ELU()( hidden )
    with K.name_scope('Layer2') :
        hidden = Conv2D(24, 3)( hidden )
        print( 'Layer 2:', hidden.shape )
        hidden = MaxPooling2D(pool_size=(2, 2))( hidden )
        hidden = ELU()( hidden )
    with K.name_scope('Layer3') :
        hidden = Conv2D(36, 3)( hidden )
        print( 'Layer 3:', hidden.shape )
        hidden = MaxPooling2D(pool_size=(2, 2))( hidden )
        hidden = ELU()( hidden )
    hidden = Dropout(.5)( hidden )
    hidden = Flatten()( hidden )
    with K.name_scope('Layer4') :
        hidden = Dense(512)( hidden )
        hidden = ELU()( hidden )
    #  with K.name_scope('Layer5'):\n",
    #      model.add(Dense(512))\n",
    #      #  model.add(BatchNormalization())\n",
    #      #  model.add(Dropout(.3))\n",
    #      model.add(ELU()) # model.add(Activation('relu'))\n",
    with K.name_scope('Output') :
        #  model.add(Dropout(.3))
        hidden = Dense(n_classes)( hidden )
        predictions = Activation('softmax')( hidden )
    model = Model(inputs=input_, outputs=predictions)
    return model
def todo2():
    from keras.optimizers import Adam
    from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


    # reproducible results :)
    np.random.seed(0)


    input_shape = X_train[0,...].shape



    # create model
    model = create_model( input_shape )

    # train model

    # generators
    train_generator = create_train_generator( train_fns, train_y )
    validation_generator = create_validation_generator( test_fns, test_y )

    # training meta
    batch_size = 32
    optimizer = Adam(lr=1e-4)
    # callbacks
    log_dir = '/tmp/log_dir'
    logging = TensorBoard( log_dir=log_dir )
    reduce_lr = ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=2, verbose=1 )
    early_stopping = EarlyStopping( monitor='val_loss', min_delta=0, patience=5, verbose=1 )

    # train
    model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'] )
    history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=validation_generator,
            validation_steps=max(1, num_val//batch_size),
            epochs=30,
            initial_epoch=0,
            callbacks=[logging, reduce_lr, early_stopping]
            )

    # save
    fn = '/tmp/model.h5'
    model.save( fn )

    # eval model
    score = model.evaluate( X_test, Y_test, verbose=1 )
    print('Test score (loss?):', score[0])
    print('Test accuracy:', score[1])










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
