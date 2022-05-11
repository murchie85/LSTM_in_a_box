# LSTM in a box

  
![](archive/box.png)


The idea is to abstract out as much functionality to configurable values to make a lite deployable service in a docker container. 

[`status`] In Progress.




# MAC M1 requirements NEED SPECIAL CONDA ENVIRONEMNT 


(Lots of conflicts and too much work - so using seperate env gets round this )
Also need to run `python -m spacy download en_core_web_sm`

- conda install spacy
- `python -m spacy download en_core_web_sm`
- conda install pandas 
- conda install keras
- conda install tensorflow==1.15
