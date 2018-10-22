#!/usr/bin/env python3
"""
"""



import numpy as np
import os
import matplotlib.pyplot as plt
import cv2



fn = 'data.csv'
import pandas as pd
col_names = ['label', 'band1_m', 'band2_m', 'band3_m', 'band4_m']
df = pd.read_table( fn, delimiter=',', header=None, names=col_names )
print( df.head() )
print( df.label.value_counts() )


#  from sklearn.decomposition import FactorAnalysis
#  feat_cols = col_names[1:]
#  FA = FactorAnalysis(n_components = 2).fit_transform( df[feat_cols].values )


from sklearn.decomposition import PCA
feat_cols = col_names[1:]
pca = PCA( n_components=len(feat_cols) )
pca_result = pca.fit_transform( df[feat_cols].values )
plt.plot( range(4), pca.explained_variance_ratio_ )
plt.plot( range(4), np.cumsum(pca.explained_variance_ratio_) )
plt.title( 'Component-wise and Cumulative Explained Variance' )
plt.show()


#  def _main() :
#      pass
#  
#  if __name__ == '__main__':
#      _main()
