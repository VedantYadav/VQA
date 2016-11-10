# Visual Question Answering

This is a keras implementation of VIS+LSTM and 2-VIS+LSTM models for the task of Visual Question Answering. These models are explained in the paper [Exploring Models and Data for Image Question Answering](https://arxiv.org/abs/1505.02074). Details about the dataset are explained at the [VisualQA website](http://www.visualqa.org/). 

## Requirements

* Python 2.7
* Numpy
* Scipy (for loading pre-computed MS COCO features)
* NLTK (for tokenizer)
* Keras
* Theano

## Training

* The basic usage is `python train.py`. 

* The model can be specified using the option `-model`. For example, to train the VIS+LSTM model enter `python train.py -model=1`. Similarly, the 2-VIS+LSTM model can be trained using `python train.py -model=2`. If no model is specified, model 1 is selected.

* The batch size and the number of epochs can also be specified using the options `-num_epochs` and `-batch_size`. The default batch size and number of epochs are 200 and 25 respectively.

* Performance of both models on the validation set is as follows:

| Model      | Epochs | Batch Size | Validation Accuracy |
|------------|--------|------------|---------------------|
| VIS+LSTM   | 10     | 200        | 54%                 |
| 2-VIS+LSTM | 10     | 200        | 53%                 |

## Prediction

* Prediction can be performed on any image using the script `question_answer.py`. The options `-image` and `-question` are used to specify the address of the image and the question respectively. 

Here are some examples of predictions using the 2-VIS+LSTM model.

| Image                                              | Question                   | Top Answers (left to right) |
|----------------------------------------------------|----------------------------|-----------------------------|
| <img src="examples/COCO_val2014_000000000136.jpg"> | Which animal is this?      | giraffe, cat, bear          |
| <img src="examples/COCO_val2014_000000000073.jpg"> | Which vehicle is this?     | motorcycle, taxi, train     |
| <img src="examples/COCO_val2014_000000000196.jpg"> | How many dishes are there? | 5, 3, 2                     |
| <img src="examples/COCO_val2014_000000000283.jpg"> | What is in the bottle?     | water, beer, wine           |
| <img src="examples/COCO_val2014_000000000357.jpg"> | Which sport is this?       | tennis, baseball, frisbee   |

