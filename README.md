# Visual Question Answering

This is a keras implementation of VIS+LSTM and 2-VIS+LSTM models for the task of Visual Question Answering. These models are explained in the paper [Exploring Models and Data for Image Question Answering](https://arxiv.org/abs/1505.02074). Details about the dataset are explained at the [VisualQA website](http://www.visualqa.org/). 

## Requirements

* Python 2.7
* Numpy
* Scipy(for loading pre-computed MS COCO features)
* NLTK(for tokenizer)
* Keras
* Theano

## Training

* The basic usage is `python train.py`. 

* The model can be specified using the option `-model`. For example, to train the VIS+LSTM model enter `python train.py -model=1`. Similarly, the 2-VIS+LSTM model can be trained using `python train.py -model=2`. If no model is specified, model 1 is selected.

* The batch size and the number of epochs can also be specified using the options `-num_epochs` and `-batch_size`. The default batch size and number of epochs are 200 and 25 respectively.

## Prediction

* Prediction can be performed on any image using the script `question_answer.py`. The options `-image` and `-question` are used to specify the address of the image and the question respectively. 
