import tensorflow as tf
from tensorflow.contrib import rnn
from models.net_utils import get_init_embedding


class NaiveRNN(object):
    def __init__(self, reversed_dict, document_max_len, num_class, args):
        self.vocabulary_size = len(reversed_dict)
        self.embedding_size = args.embedding_size
        self.num_hidden = args.num_hidden
        self.num_layers = args.num_layers
        self.learning_rate = args.learning_rate

        self.x = tf.placeholder(tf.int32, [None, document_max_len], name="x")
        self.x_len = tf.reduce_sum(tf.sign(self.x), 1)
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("embedding"):
            if args.glove:
                init_embeddings = tf.constant(get_init_embedding(reversed_dict, self.embedding_size), dtype=tf.float32)
                self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)
            else:
                init_embeddings = tf.random_uniform([self.vocabulary_size, self.embedding_size])
                self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=True)
            self.x_emb = tf.nn.embedding_lookup(self.embeddings, self.x)

        with tf.name_scope("birnn"):
            fw_cells = [rnn.BasicLSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            bw_cells = [rnn.BasicLSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in bw_cells]

            self.rnn_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.x_emb, sequence_length=self.x_len, dtype=tf.float32)
            self.last_output = self.rnn_outputs[:, -1, :]

        with tf.name_scope("output"):
            self.logits = tf.contrib.slim.fully_connected(self.last_output, num_class, activation_fn=None)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
