# Text Data Preprocessing Library
# NLTK is also known as the Natural Language Toolkit. 
# It is the Python library used for preprocessing text data. 
# It has methods for cleaning the data and removing repetitive words.
import nltk
#nltk.download('punkt')

# This class is responsible to give the stem words for given words.
from nltk.stem import PorterStemmer
# defining object for the PorterStemmer() class
stemmer = PorterStemmer()

# reading and processing JSON data.
import json

# Pickle is the Python library that converts lists, dictionaries, and other objects into streams of zero and one. 
# This will be helpful to store preprocessed training data.
import pickle

# training dataset has to be Numpy arrays.
import numpy as np

words=[]
classes = []
word_tags_list = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

# function for appending stem words
# contain all the words converted into their stem words and the punctuation will be removed.
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)  
    return stem_words



for intent in intents['intents']:
    
        # Add all words of patterns to list
        for pattern in intent['patterns']:     

            # Tokenize each pattern and store it in the pattern_word variable.       
            pattern_word = nltk.word_tokenize(pattern)    
            #print("pattern_word: ",pattern_word)  

            # The extend() method will add all the tokenized patterns into the list words.      
            words.extend(pattern_word) 
            # print("words:",words)     

            # This is an empty list that is appended by words and tags.                
            word_tags_list.append((pattern_word, intent['tag']))
            #print("word_tags_list: ",word_tags_list)   

        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            #print("classes: ",classes) 

            # call the get_stem_words() function to create the list of stem words while excluding ignore_words.
            stem_words = get_stem_words(words, ignore_words)

#print(" stem_words  1: ",stem_words)
#print("word_tags_list[0] :  ",word_tags_list[0]) 
#print("classes :  ",classes)   

#####################################################  2   ##############################################################

#Create word corpus for chatbot
def create_bot_corpus(stem_words, classes):

    # List of stem_words and classes are converted into set to get unique words from these lists. 
    # Again they are converted into list and stored in sorted manner.
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    # Create two pickle files by name words.pkl and classes.pkl. 
    # These are then written by pickle.dump() method for creating the training dataset. 
    # ‘wb’ stands for write in binary mode. Thus the data is stored in the binary form(0 and 1) in these files.
    # Note: These files will be created automatically in the same folder when you run the code.
    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(classes, open('classes.pkl','wb'))

    # this function returns the sorted stem_word list and sorted list of classes.
    return stem_words, classes

stem_words, classes = create_bot_corpus(stem_words,classes)  

#print(" stem_words  2: ", stem_words)
#print("classes :  ",classes)

####################################################  3   ##############################################################

# Define an empty list by the name training_data.
training_data = []

# Define number_of_tags by finding the length of classes list.
number_of_tags = len(classes)
#print("number_of_tags : ", number_of_tags )

# Define an array of zeroes by name labels. Labels will store the corresponding tag of the sentence.
labels = [0]*number_of_tags
#print("labels : ", labels )

# Create bag of words and labels_encoding
for word_tags in word_tags_list:
        
        bag_of_words = []       
        pattern_words = word_tags[0]
       
        for word in pattern_words:
            # index() method is a built-in function used to find the index of the first occurrence of a specified value in a list.
            index=pattern_words.index(word)
            #print("index: ",index)

            # word.lower() is first converting the word to lowercase 
            # stemmer.stem() is used to find the stem or root word
            word=stemmer.stem(word.lower())

            #root word is replaced with the orginal word
            pattern_words[index]=word  
            #print("pattern_words: ",pattern_words)

        # if these words are present in the stem words, 1 is appended in bag_of_words else 0 is appended.
        for word in stem_words:
            if word in pattern_words:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)

        #print("stem words: ", stem_words)        
        #print("bag_of_words:  ", bag_of_words)
                
        # creating label encoding        

        labels_encoding = list(labels) #labels all zeroes initially
        tag = word_tags[1] #save tag
        tag_index = classes.index(tag)  #go to index of tag
        labels_encoding[tag_index] = 1  #append 1 at that index
        #print("labels_encoding : ", labels_encoding)

        training_data.append([bag_of_words, labels_encoding])

#print("training_data[0] : ",training_data[0])

########################################################  4   #########################################################


# Create training data
def preprocess_train_data(training_data):
   
    training_data = np.array(training_data, dtype=object)
    
    train_x = list(training_data[:,0])
    train_y = list(training_data[:,1])

    print(train_x[0])
    print(train_y[0])
  
    return train_x, train_y

train_x, train_y = preprocess_train_data(training_data)




