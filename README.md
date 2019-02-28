# Semantic textual similarity using string similarity
### Jin Young Yang
### Feb 20 2019
<br>


### String Similarity Metrics

*   BLEU -  It is an algorithm for evaluating the quility of text and its value indicates how similar the candidate text is to the reference texts, with values closer to 1 representing more similar texts.
*   NIST -  Where BLEU simply calculates n-gram precision adding equal weight to each one, NIST also calculates how informative a particular n-gram is. The rarer that n-gram is, the more weight it will be given.
*   Levenshtein Distance -  It measures the difference between two sequences such as the minimum number of single-character edits required to change one word into the other. 
*   Longest Common String -   It is the longest common string between two sequences including any special charater and space. 
*   Word Error rate -   Derived from the Levenshtein distance, it works at the word level instead of the phoneme level, compares different systems, and evaluates improvements within one system.
<br>

### pi_logreg.py   Description
<br>
**load_X function**

In general, it opens `sts-test.csv` file and find the string similarity metrics to acheive the pearsons correlation scores which is then stored in an ouput file `test_output.txt`.

For each line of the input, the two comparing sentences are stored as `t1` and `t2` and their tokenized word lists as `token1` and `token2`, respectively.
The two sentences are then used to find the five string similarity matrics that we are interested in. 

For BLEU and NIST scores, we need to sum the matrics in both order in order to count the case when the length of the sentences are different. Since NIST has some zero denominator, `nist_func(x,y)` is built in order to save the zero denominator as zero. 

For Levenshtein distance, the main function uses `edit_distance` function in `nltk` package. Word error rate is then calculated with the tokenized sentences - the sum of the edit distance of tokenized sentences divided by length of the first tokenized sentence and that divided by the second tokenized sentence - in order to count the different number of words in the two comparing sentences.

Finally, the main function uses the built-in function `findCommonString(s, lowerLengthString)` in order to find the longest common substring in the two sentences. `findCommonString` first sets the longer sentence as the first argument and find the longest common substring using a couple of loops. 

After receiving all matrics for each line (appended in an appropriate empty list each time), the main function then calculates the pearsons correlation coefficient using `pearsonr` function in `scipy`. 

There scores are then reported to the out file, `test_output.txt`
<br>

**main function**

With the mininum paraphrase and maximum paraphrase, the main function calls the train and dev datasets and uses logistic regression model from sklearn import. We train the model with train dataset and get the score of dev datasets from this trained logistic regression model.
<br>

**built-in function**

* `nist_func(x,y)` catches the zero denominator and save it as zero to solve the error `ZeroDivisionError`
* `findCommonString(s, lowerLengthString)` sets the first argument as the longer one and find the longest common substring using the index of the string by character. It also counts any special character and space in between the words
* `load_sts(sts_data)` simply opens the input file and creates text and label lists
* `sts_to_pi(texts, labels, min_paraphrase=4.0, max_nonparaphrase=3.0)` uses its parameter to limit the dataset as paraphrase.


###  Command Line Flags
`python pi_logreg.py`

### Result

This system scores 0.8493038493038493.
