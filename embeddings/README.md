## Instructions for preparing the embedding matrix.

Download and extract the pretrained common crawl 300D word vectors from http://nlp.stanford.edu/projects/glove/ .
Use the function `create` from `embedding.py` to generate the files `embedding_matrix.h5` and `word_idx`. The usage is as follows:

```
>> import embedding
>> embedding.create('address-of-extracted-file')
```
