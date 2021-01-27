
import pandas as pd
from glob import glob
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from scipy.stats import ttest_1samp, chi2_contingency, ks_2samp, ttest_ind, ttest_rel
from tqdm import tqdm
import numpy as np
import warnings, sys, getopt
warnings.filterwarnings('ignore')

def aggregate(inpath, outpath):

	fnames = glob(inpath+'/*.tsv')
	results = []

	for fname in tqdm(fnames):
		print(fname)
		label = fname.split('/')[-1].replace('.tsv', '')

		df = pd.read_csv(fname, sep='\t')

		df['label'] = label

		no_convergence = len(df[df['warning']=='ConvergenceWarning']['fold'].value_counts())
		
		n = len(df)*1.0/20

		if (len(df)==0) or (len(df['truth'].value_counts())==1):
			print(fname)
			results.append((label, n, no_convergence,\
			 None, None, None, None, None,\
			 None, None, None, None, None, None, \
			 None, None))
			continue

		minority_label = df['truth'].value_counts().index[1]
		pct_minority = (df['truth'].value_counts()[minority_label]*1.0)/len(df)

		if no_convergence==20:
			print(fname)
			results.append((label, n, no_convergence,\
			 pct_minority, 20, 20, 1-pct_minority, 1-pct_minority,\
			 1-pct_minority, 1-pct_minority, 1-pct_minority, 1-pct_minority, 1-pct_minority, 1-pct_minority, \
			 0, 99))
			continue

		df = df[df['warning']!='ConvergenceWarning']

		df['monkey'] = df['monkey'].astype('bool')
		df['my_preds'] = df['my_preds'].astype('bool')

		grobj = df.groupby('fold')

		my_f1_dist 		  = grobj.apply(lambda x: f1_score(x['truth'], x['my_preds'], average='binary'))#,# pos_label=minority_label))
		monkey_f1_dist 	  = grobj.apply(lambda x: f1_score(x['truth'], x['monkey'],   average='binary'))#, pos_label=minority_label))

		my_acc 		= accuracy_score(df['truth'], df['my_preds'])
		monkey_acc  = accuracy_score(df['truth'], df['monkey'])

		my_recall 	  = recall_score(df['truth'], df['my_preds'], average='binary')#, pos_label=minority_label)
		monkey_recall = recall_score(df['truth'], df['monkey'], average='binary')#, pos_label=minority_label)

		my_precision 	 = precision_score(df['truth'], df['my_preds'], average='binary')#, pos_label=minority_label)
		monkey_precision = precision_score(df['truth'], df['monkey'], average='binary')#, pos_label=minority_label)

		my_maj_cnt = grobj.apply(lambda x: len(x['my_preds'].value_counts())==1).sum()
		monkey_maj_cnt = grobj.apply(lambda x: len(x['monkey'].value_counts())==1).sum()

		my_f1 		= my_f1_dist.mean()
		monkey_f1 	= monkey_f1_dist.mean()

		stat, pval = ttest_1samp(my_f1_dist, monkey_f1_dist.mean())

		results.append((label, n, no_convergence, pct_minority, my_maj_cnt, monkey_maj_cnt, my_recall, monkey_recall,\
			 my_precision, monkey_precision, my_acc, monkey_acc, my_f1, monkey_f1, \
			stat, pval))

	df = pd.DataFrame(results)
	df.columns = ['label', 'n', 'no_convergence', 'pct_minority', 'my_maj_cnt', 'monkey_maj_cnt',\
			'my_recall', 'monkey_recall', 'my_precision', 'monkey_precision', 'my_acc', 'monkey_acc',\
			 'my_f1', 'monkey_f1', 'stat', 'pval']#, 'stat2', 'pval2', 'stat3', 'pval3', 'stat4', 'pval4']

	df.sort_values('stat', inplace=True)
	df['rank'] = range(1, len(df)+1)

	df.to_csv(outpath)


def main(argv):
    inpath, outpath = '', ''

    try:
      opts, args = getopt.getopt(argv,"h:i:o:",\
        ["inpath=","outpath="])
    except getopt.GetoptError:
      print('aggregate.py -i <inputpath> -o <outputspath>')
      sys.exit(2)

    for opt, arg in opts:
      if opt == '-h':
         print('aggregate.py -i <inputpath> -o <outputspath>')
         sys.exit()
      elif opt in ("-i", "--inputpath"):
         inpath = arg
      elif opt in ("-o", "--outpath"):
         outpath = arg

    print('Input path is ', inpath)
    print('Output path is ', outpath)

    aggregate(inpath, outpath)

if __name__ == "__main__":
   main(sys.argv[1:])


