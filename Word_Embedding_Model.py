from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D

################ Data Preparation ######################

def getText(docpath):
    # open the file as read only
    docfile = open(docpath, 'r')
    #read all text
    text = docfile.read()
    #close file 
    docfile.close()
    return text
#
#def clean_text(text):
#    # split into tokens by white space
#    tokens = text.split()
#    # remove punctuation from each token
#    table = str.maketrans('','', punctuation)
#    tokens = [word.translate(table) for word in tokens]
#    # remove remaining tokens that are not alphabetic
#    tokens = [word for word in tokens if word.isalpha() ]
#    # filter out stop words
#    stop_words = set(stopwords.words('english'))
#    # filter out stop words
#    tokens = [word for word in tokens if not word in stop_words]
#    # filter out short tokens
#    tokens = [word for word in tokens if len(word) > 1]
#    return tokens
#
#
#def add_to_vocab(docpath, vocab):
#    #get text from document
#    text = getText(docpath)
#    # turn a text into clean tokens
#    tokens = clean_text(text)
#    # update counts
#    vocab.update(tokens)
#
#def process_docs(directory, vocab, is_train):
#    # walk through all files in the folder
#    for filename in os.listdir(directory):
#        # skip any reviews in the test set
#        if is_train and filename.startswith('cv9'):           
#            continue
#        if not is_train and not filename.startswith('cv9'):
#            continue
#        # create the full path of the file to open
#        docpath = directory + '/' + filename
#        add_to_vocab(docpath, vocab)
#    
#def save_tokens(tokens, filename):
#    data = '\n'.join(tokens)
#    file  = open(filename, 'w')
#    file.write(data)
#    file.close()
#
##define vocab
#vocab = Counter()
## add all docs to vocab
#process_docs('txt_sentoken/neg', vocab, True)
#process_docs('txt_sentoken/pos', vocab, True)
#
## keep tokens with a min occurrence
#min_occurance = 2
#tokens = [word for word,occurance in vocab.items() if occurance >= min_occurance]
#
## save tokens to a vocabulary file
#save_tokens(tokens, 'vocab.txt')


################### Train Embedding Layer ################################
    
# load the vocabulary
vocabulary = getText('vocab.txt')
vocabulary = vocabulary.split()
vocabulary = set(vocabulary)

# turn a doc into clean tokens
def clean_doc(doc, vocab):
	 #split into tokens by white space
    tokens = doc.split()
	 #remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    #filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in os.listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = getText(path)
		# clean doc
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents


# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocabulary, True)
negative_docs = process_docs('txt_sentoken/neg', vocabulary, True)
train_docs = negative_docs + positive_docs


# create Tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)


# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = np.array([0 for _ in range(900)] + [1 for _ in range(900)])

# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocabulary, False)
negative_docs = process_docs('txt_sentoken/neg', vocabulary, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = np.array([0 for _ in range(100)] + [1 for _ in range(100)])

# define vocabulary size
vocab_size = len(vocabulary) + 1

# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#print(model.summary())

# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)

# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
#
ytest_predicted = model.predict_classes(Xtest)




