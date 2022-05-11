import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join



"""
returns an array of letters 
"""
def letterArrayFromTextFiles(mypath,printme=True):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    data = ''
    fullData = ''
    fileList = []
    for file in onlyfiles:
        if file[-3:] == 'txt':
            try:
                with open(str(mypath) + '/' + str(file)) as f:
                    data = f.read().replace('\n', ' ')
                if(printme):
                    print('{} : imported succesfully.'.format(file))
            except:
                if(printme):
                    print('{} : import **failed**.'.format(file))
                
                
        fullData = fullData + data
    
    if(printme):
        print('Length of Data is: ' + str(len(fullData)))
        print('Full Data Preview: ' + str(fullData[:100]))
        print('-----------------------------------\n\n')
    
    return(fullData)








# EXTRACT TOKENS 
def get_tokens(doc_text,nlp,printme=True):
    # This pattern is a modification of the defaul filter from the
    # Tokenizer() object in keras.preprocessing.text. 
    # It just indicates which patters no skip.
    skip_pattern = '\r\n \n\n \n\n\n!"-#$%&()--.*+,-./:;<=>?@[\\]^_`{|}~\t\n\r '
    
    tokens = [token.text.lower() for token in nlp(doc_text) if token.text not in skip_pattern]
    
    if(printme):
        print('Token sample: ' + str(tokens[0:9]))
        print('Length of processed tokens are : ' + str(len(tokens) ))
        print('-----------------------------------\n\n')
    
    return tokens






#--------------------------------------------------------
#              GENERATES TEXT WITH FITTED MODEL         #
#--------------------------------------------------------



from keras.preprocessing.sequence import pad_sequences 

def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    # List to store the generated words. 
    predictedText = []
    # Set seed_text as input_text. 
    input_text = seed_text
    
    for i in range(num_gen_words):
        
        # Encode input text. 
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        
        # Add if the input tesxt does not have length len_0.
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        
        # Do the prediction. Here we automatically choose the word with highest probability. 
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        
        # Convert from numeric to word. 
        pred_word = tokenizer.index_word[pred_word_ind]
        
        # Attach predicted word. 
        input_text += ' ' + pred_word
        
        # Append new word to the list. 
        predictedText.append(pred_word)
        
    return ' '.join(predictedText)





# Trying again with various temp settings 


def generate_textWithTemperature(model, tokenizer, seq_len, seed_text, num_gen_words, temperature):
    
    predictedText = []
    
    input_text = seed_text
    
    for i in range(num_gen_words):
        
        # Encode input text. 
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        
         # Add if the input tesxt does not have length len_0.
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        
        # Get learned distribution.
        pred_distribution = model.predict(pad_encoded, verbose=0)[0]
        
        # Apply temperature transformation.
        new_pred_distribution = np.power(pred_distribution, (1 / temperature)) 
        new_pred_distribution = new_pred_distribution / new_pred_distribution.sum()
        
        # Sample from modified distribution.
        choices = range(new_pred_distribution.size)
 
        pred_word_ind = np.random.choice(a=choices, p=new_pred_distribution)
        
        # Convert from numeric to word. 
        pred_word = tokenizer.index_word[pred_word_ind]
        
        # Attach predicted word. 
        input_text += ' ' + pred_word
        
        # Append new word to the list. 
        predictedText.append(pred_word)
    return ' '.join(predictedText)
        

