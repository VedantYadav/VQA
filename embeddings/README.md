## Instructions for preparing embeddings

Download and extract the pretrained common crawl 300D word vectors from http://nlp.stanford.edu/data/glove.840B.300d.zip.
Use the function `create` from `embedding.py` to generate the files `embedding_matrix.h5` and `word_idx`. The usage is as follows:

Change to the VQA directory and on the python command line execute the following commands

```
>> import embedding
>> embedding.create('address-of-extracted-glove-file')
```
