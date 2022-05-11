# LSTM in a box

  
![](archive/box.png)
*LSTM that can be deployed in a docker container and configured to run on your data*
    
[`status`] In Progress.  

## Basics  
  

- A LSTM that you point to your text
- It trains inside a docker container
- Model is saved locally
- Prediction is provided at end of run.  


The idea is to abstract out as much functionality to configurable values to make a lite deployable service in a docker container. 


## LAYERED LSTM Script Generator
  
1. Import dependencies
2. Append all files to name array
    2.1 Append all file contents to string.
3. Load Spacy language model 
4. Extract tokens with manual function.  
5. Generate sequence array, with each element being 26 words, the following elemenet moved along by one word only. 
6. Import Keras tokeniser and `fit_on_texts` on sequence array. 
    6.1 Apply `texts_to_sequences` to the same sequence array.
    6.2 Verify fitted correctly by checking index 
7. Pass sequence to np array 
8. Construct observation matrix 
9. Define X and y values (25 tokens and 26th target token)
10. Define LSTM multi layer model with embeddings, softmax and relu 
11. Create model 
12. Fit model 
13. Save tokeniser and model
14. Generate new text
    14.1 Try existing seed seed
    14.2 Try with brand new seed
15. Repeat with various temperature values





# MAC M1 requirements NEED SPECIAL CONDA ENVIRONEMNT 


(Lots of conflicts and too much work - so using seperate env gets round this )
Also need to run `python -m spacy download en_core_web_sm`

- conda install spacy
- `python -m spacy download en_core_web_sm`
- conda install pandas 
- conda install keras
- conda install tensorflow==1.15
