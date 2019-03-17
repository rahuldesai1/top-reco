#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import keras
import pickle

print("Finished importing dependencies...")

# Use header to load my csv file with variable names
header = ['label','rndm','weight']
# boosted frame
header += ['W_B_pt','W_B_eta','W_B_phi','W_B_E','W_B_m']
header += ['b_B_pt','b_B_eta','b_B_phi','b_B_E','b_B_m']
header += ['Wjj_B_dR','tWb_B_dR'] #dR = sqrt(phi^2 + eta^2)
header += ['Wjj_B_deta','Wjj_B_dphi','tWb_B_deta','tWb_B_dphi']
# lab frame
header += ['W_pt','W_eta','W_phi','W_E','W_m']
header += ['b_pt','b_eta','b_phi','b_E','b_m']
header += ['Wjj_dR','tWb_dR']
header += ['Wjj_deta','Wjj_dphi','tWb_deta','tWb_dphi']
header += ['t_pt','t_eta','t_phi','t_E','t_m']

header += ['btag1', 'btag2', 'btag3'] #binary representation of likelihood of the jet being an actual bjet

print("Loading Data...")
df = pd.read_csv('~/projects/top-reco-tests/samples/result.csv', names=header, delimiter=' ', skiprows=1)
print(df.shape)
pos_class = df[df['label'] == 1]
neg_class = df[df['label'] == 0]
neg_class = neg_class.sample(frac=0.1)
neg_class.shape
final_df = pd.concat([neg_class, pos_class])

y = final_df['label']
X = final_df.drop('label', axis=1).drop('rndm', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.15, shuffle=True)

print("Performing PCA...")
pca = PCA(n_components=25)
X_new = pca.fit_transform(X_train)
new_df = pd.DataFrame(X_new)
new_df.head()
print("Done")

print("Trianing the SVM...")
classifier = SVC()
classifier.fit(new_df, y_train)
print("Done.")

X_test_new = pd.DataFrame(pca.transform(X_test))
print(classifier.score(X_test_new, y_test))

#save the pca and svm models
print("Saving the models")
pickle.dump(pca, open('models/pca_model.sav', 'wb'))
pickle.dump(classifier, open('models/svm_model.sav', 'wb'))
print("Done.")
