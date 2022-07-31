# Perceptron-Message-Spam-Classifier

The data set is divided into two sets: training set and test set. Each set has two directories: spam and ham. All files in the spam folders are spam messages and all files in the ham folder are legitimate (non spam) messages.
The Perceptron algorithm described here: https://www.cs.utah.edu/~zhe/pdf/lec-10-perceptron-upload.pdf 

Ignore punctuation and special characters and normalize words by converting them to lower case, converting plural words to singular (i.e., \Here" and \here" are the same word, \pens" and \pen" are the same word). After that this Perceptron Algorithm will improve by throwing away (i.e., ltering out) stop words such as \the" \of" and \for" from all the documents.
 
Compiling command is below

python main.py train test (number of iterations) (learning rate)


