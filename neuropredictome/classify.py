
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from glob import glob
from multiprocessing import Pool
import sys, getopt, warnings, traceback
import pandas as pd
import numpy as np
import re

warnings.filterwarnings('error')

def classify(X, ys, label, outpath):

    y = ys[label].dropna().astype(bool)

    common_subjs = X.index.intersection(y.index)
    X_label = X.loc[common_subjs]
    y = y.loc[common_subjs]

    X_monkey = X_label.sample(frac=1)
    X_monkey.index = X_label.index
    X_monkey[['age', 'gender']] = X_label[['age', 'gender']]

    features = X_label.columns.drop(['age', 'gender'])

    all_preds = pd.DataFrame()
    weights_mine = []
    weights_monkey = []

    for fold in range(20):

        try:
            X_train, X_test, y_train, y_test = train_test_split(X_label, y)

            preds            	 = pd.DataFrame()
            preds['truth']   	 = y_test
            preds['gender']  	 = X_test['gender']
            preds['age']     	 = X_test['age']
            preds['monkey']  	 = 'NA'
            preds['my_preds'] 	 = 'NA'
            preds['fold']		 = fold
            preds['warning']     = None
            preds['my_preds_prob'] = None
            preds['monkey_prob'] = None

            clf_mine = LogisticRegression(solver='lbfgs', max_iter=5000)
            clf_monkey = LogisticRegression(solver='lbfgs', max_iter=5000)

            clf_mine.fit(X_train, y_train)
            clf_monkey.fit(X_monkey.loc[X_train.index], y_train)

            preds['my_preds'] = clf_mine.predict(X_test)
            preds['monkey']   = clf_monkey.predict(X_test)

            preds['my_preds_prob'] = clf_mine.predict_proba(X_test)[:,0]
            preds['monkey_prob']   = clf_monkey.predict_proba(X_test)[:,0]
            all_preds = pd.concat([all_preds, preds])

            weights_monkey.append(clf_monkey.coef_[0])
            weights_mine.append(clf_mine.coef_[0])
        except ConvergenceWarning:
            preds['warning'] = 'ConvergenceWarning'
            all_preds = pd.concat([all_preds, preds])    

    label = label.replace('/', '_')
    all_preds.to_csv('%s/%s.tsv' % (outpath, label), sep='\t')
    pd.DataFrame(weights_mine).to_csv('%s/%s_weights_mine.csv' % (outpath, label))
    pd.DataFrame(weights_monkey).to_csv('%s/%s_weights_monkey.csv' % (outpath, label))

def process(X, ys, label, outpath):
    try:
        n_preds = classify(X, ys, label, outpath)
    except Exception as e:
        label = label.replace('/', '')
        f = open('%s/exceptions_%s.txt' % (outpath, label), 'w')
        print(str(e)+'\n')
        f.write(str(label)+'\n')
        f.write(str(e)+'\n')
        f.write(traceback.format_exc())
        f.close()

def main(argv):
    features, labels, outpath = '', '', ''

    try:
      opts, args = getopt.getopt(argv,"h:x:y:o:",\
        ["features=","labels=","outpath="])
    except getopt.GetoptError:
      print('classify.py -x <featuresfile> -y <labelsfile> -o <outputspath>')
      sys.exit(2)

    for opt, arg in opts:
      if opt == '-h':
         print('classify.py -x <featuresfile> -y <labelsfile> -o <outputspath>')
         sys.exit()
      elif opt in ("-x", "--features"):
         features = arg
      elif opt in ("-y", "--labels"):
         labels = arg
      elif opt in ("-o", "--outpath"):
         outpath = arg

    print('Features file is ', features)
    print('Labels file is ', labels)
    print('Output file is ', outpath)

    X = pd.read_csv(features, index_col=0)
    ys = pd.read_csv(labels, index_col=0)

    for col in ys.columns:
        process(X, ys, col, outpath)

if __name__ == "__main__":
   main(sys.argv[1:])

