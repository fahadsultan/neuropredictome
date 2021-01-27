
import warnings, sys, getopt
import pandas as pd 
from classify import process
from aggregate import aggregate

def main(argv):
    features, labels, outpath = '', '', ''

    try:
      opts, args = getopt.getopt(argv,"h:x:y:o:",\
        ["features=","labels=","outpath="])
    except getopt.GetoptError:
      print('neuropredictome.py -x <featuresfile> -y <labelsfile> -o <outputspath>')
      sys.exit(2)

    for opt, arg in opts:
      if opt == '-h':
         print('neuropredictome.py -x <featuresfile> -y <labelsfile> -o <outputspath>')
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

    aggregate(outpath, outpath+'/results.csv')

if __name__ == "__main__":
   main(sys.argv[1:])

