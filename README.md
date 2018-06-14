# text-classification-tensorflow
  Tensorflow implementation of **Attention-based Bidirectional RNN** text classification. (with naive Bidirectional RNN model as a baseline)
  <img src="https://user-images.githubusercontent.com/6512394/41424160-42520358-7038-11e8-8db0-859346a1fa3a.PNG">


## Requirements
- Python 3
- Tensorflow
- pip install -r requirements.txt


## Usage

### Prepare Data
We are using pre-processed version of [Twitter Sentiment Classification Data](http://help.sentiment140.com/for-students). To use sample data (100K train/30K test),
```
$ unzip sample_data/sample_data.zip -d sample_data
```

To use full data (1.2M train/0.4M test), download it from [google drive link](https://drive.google.com/file/d/1aMt-6OCN_mEDlmRX4bymk5ZNEatsVXF-/view?usp=sharing).

To use Glove pre-trained embedding, download it via
```
$ python download_glove.py
```

### Train
To train the model with sample data,
```
$ python train.py
```
Train data is split to train set(85%) and validation set(15%). Every 2000 steps, the classification accuracy is tested with validation set and the best model is saved.


To use Glove pre-trained vectors as initial embedding,
```
$ python train.py --glove
```

#### Additional Hyperparameters
```
$ python train.py -h
usage: train.py [-h] [--train_tsv TRAIN_TSV] [--model MODEL] [--glove]
                [--embedding_size EMBEDDING_SIZE] [--num_hidden NUM_HIDDEN]
                [--num_layers NUM_LAYERS] [--learning_rate LEARNING_RATE]
                [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
                [--keep_prob KEEP_PROB] [--checkpoint_dir CHECKPOINT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --train_tsv TRAIN_TSV
                        Train tsv file.
  --model MODEL         naive | att
  --glove               Use glove as initial word embedding.
  --embedding_size EMBEDDING_SIZE
                        Word embedding size. (For glove, use 50 | 100 | 200 | 300)
  --num_hidden NUM_HIDDEN
                        RNN Network size.
  --num_layers NUM_LAYERS
                        RNN Network depth.
  --learning_rate LEARNING_RATE
                        Learning rate.
  --batch_size BATCH_SIZE
                        Batch size.
  --num_epochs NUM_EPOCHS
                        Number of epochs.
  --keep_prob KEEP_PROB
                        Dropout keep prob.
  --checkpoint_dir CHECKPOINT_DIR
                        Checkpoint directory.
```



### Test
To test classification accuracy for test data,
```
$ python test.py
```

To use custom data,
```
$ python test.py --test_tsv=<CUSTOM_TSV>
```

### Sample Test Results
Trained and tested with [full data](https://drive.google.com/file/d/1aMt-6OCN_mEDlmRX4bymk5ZNEatsVXF-/view?usp=sharing) with default hyper-parameters,

Model    | Naive   | Naive(+Glove) | Attention | Attention(+Glove)
:---:    | :---:   | :---:         | :---:     | :---:
Accuracy | 0.574   | 0.578         | 0.811     | 0.820


## References
- [Dataset](http://help.sentiment140.com/for-students)
- [dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
- [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://www.aclweb.org/anthology/P16-2034)
