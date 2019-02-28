import argparse
import numpy as np
from nltk.metrics.distance import edit_distance
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import sentence_nist
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher  
from sklearn.linear_model import LogisticRegression



def load_sts(sts_data):
    """Read a dataset from a file in STS benchmark format"""
    texts = []
    labels = []

    with open(sts_data, 'r') as dd:
        for line in dd:
            fields = line.strip().split("\t")
            labels.append(float(fields[4]))
            t1 = fields[5].lower()
            t2 = fields[6].lower()
            texts.append((t1,t2))

    labels = np.asarray(labels)
    
    return texts, labels


def sts_to_pi(texts, labels, min_paraphrase=4.0, max_nonparaphrase=3.0):
    """Convert a dataset from semantic textual similarity to paraphrase.
    Remove any examples that are > max_nonparaphrase and < min_nonparaphrase.
    labels must have shape [m,1] for sklearn models, where m is the number of examples"""
    pi_rows = np.where(np.logical_or(labels>=min_paraphrase, labels<=max_nonparaphrase))[0] 
    # fine the row with correct index from above
    texts = [texts[i] for i in pi_rows]
    
    # remain the desired labels
    pi_y = labels[pi_rows]
    
    # result is boolean (1/true : paraphrase, 0/false : nonparaphrase)
    labels = pi_y > max_nonparaphrase 
    return texts, labels

def nist_func(x,y):
    "catch the zero dividend and return it as zero"
    try:
        nist1 = sentence_nist([x],y)
        nist2 = sentence_nist([y],x)
        return nist1+nist2
    except ZeroDivisionError:
        return 0

def findCommonString(x,y):
    match = SequenceMatcher(None, x,y).find_longest_match(0, len(x), 0, len(y))
    common_str=x[match.a: match.a + match.size]
    return (len(common_str))


def load_X(sent_pairs):
    """Create a matrix where every row is a pair of sentences and every column in a feature.
    Feature (column) order is not important to the algorithm."""

    features = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Levenshtein distance"]

    LevenDist = []
    bleu = []
    nist = []
    error = []
    com_str = []
    

    for pair in sent_pairs:
        # two sentences
        t1, t2 = pair 
        # two tokenized sentences
        token1 = word_tokenize(t1)
        token2 = word_tokenize(t2)
        
        # find the similarity matrices
        dist = edit_distance(t1, t2)
        dist_token = edit_distance(token1, token2) # for word error rate
        bleuscore = sentence_bleu([token1],token2) + sentence_bleu([token2],token1)
        nistscore = nist_func(token1,token2)   # built-in function
        worderror = dist_token/len(token1) + dist_token/len(token2)
        long_com_str = findCommonString(t1,t2)
        
        # append the matrices for each line 
        LevenDist.append(dist)
        bleu.append(bleuscore)
        nist.append(nistscore)
        error.append(worderror)
        com_str.append(long_com_str)


    X = np.zeros((len(sent_pairs), len(features)))
    X[:,0] = nist
    X[:,1] = bleu
    X[:,2] = error
    X[:,3] = com_str
    X[:,4] = LevenDist
    return X


def main(sts_train_file, sts_dev_file):
    """Fits a logistic regression for paraphrase identification, using string similarity metrics as features.
    Prints accuracy on held-out data. Data is formatted as in the STS benchmark"""

    min_paraphrase = 4.0
    max_nonparaphrase = 3.0

    # loading train
    train_texts_sts, train_y_sts = load_sts(sts_train_file)
    train_texts, train_y = sts_to_pi(train_texts_sts, train_y_sts,
      min_paraphrase = min_paraphrase, max_nonparaphrase = max_nonparaphrase)

    train_X = load_X(train_texts)

    # loading dev
    dev_texts_sts, dev_y_sts = load_sts(sts_dev_file)
    dev_texts, dev_y = sts_to_pi(dev_texts_sts, dev_y_sts,
      min_paraphrase = min_paraphrase, max_nonparaphrase = max_nonparaphrase)

    dev_X = load_X(dev_texts)
    
    # train a logistic model using train
    logisticRegr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    logisticRegr.fit(train_X, train_y)
    
    # apply the model in dev and get the accuracy score
    score = logisticRegr.score(dev_X, dev_y)

    print(f"Found {len(train_texts)} training pairs")
    print(f"Found {len(dev_texts)} dev pairs")

    print(f"Fitting and evaluating model: the logistic regression model scores {score} on sts-dev dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_dev_file", type=str, default="sts-dev.csv",
                        help="dev file")
    parser.add_argument("--sts_train_file", type=str, default="sts-train.csv",
                        help="train file")
    args = parser.parse_args()

    main(args.sts_train_file, args.sts_dev_file)
