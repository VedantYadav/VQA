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

Both the models can be trained using the script `train.py`. To train the VIS+LSTM model enter `python train.py -model=1`. Similarly, the 2-VIS+LSTM model can be trained using `python train.py -model=2`. If no model is specified, model 1 is trained.

You can also specify the batch size and the number of epochs using `python train.py -batch_size=BATCH_SIZE -num_epochs=NUM_EPOCHS`. The default batch size is 200 and the number of epochs is 25.
