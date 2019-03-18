import pandas as pd
import numpy as np
import sys

frac = float(sys.argv[1])
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

#df = pd.read_csv('~/projects/top-reco-tests/samples/result.csv', names=header, delimiter=' ', skiprows=1)
df = pd.read_csv('~/projects/top-reco-tests/samples/result.csv', delimiter=' ', names=header, skiprows=1)

#down-sample the class of non-jet samples to 1/4 of the original size (prevents model bias towards to majority class)
pos_class = df[df['label'] == 1]
neg_class = df[df['label'] == 0]
neg_class = neg_class.sample(frac=frac)
neg_class.shape
final_df = pd.concat([neg_class, pos_class])

#dataframe preprocessing
y = df['label']
X = df.drop('label', axis=1).drop('rndm', axis=1)

#data normalization
def normalize(col):
    print(col.name)
    threshold = col.quantile(0.9)
    mini = col.min()
    slopeUpper = (1 - 0.9) / (col.max() - threshold)
    slopeLower = (0.6 - 0) / (threshold - mini)
    def norm_helper(row):
        if row > threshold:
            return 0.9 + slopeUpper * (row - threshold)
        else:
            return 0 + slopeLower * (row - mini)
    return col.apply(norm_helper)

#normalize the data
X_norm = X.apply(normalize, axis=0)

print(X_norm.head())
normalized = pd.concat([y, X_norm], axis=1)
normalized.to_csv('~/projects/top-reco-tests/samples/results_norm_{0}.csv'.format(frac), sep=',') 
