import argparse
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from models.naive_rnn import NaiveRNN
from models.attention_rnn import AttentionRNN
from data_utils import build_dict, build_dataset, batch_iter


def add_arguments(parser):
    parser.add_argument("--train_tsv", type=str, default="sample_data/train.tsv", help="Train tsv file.")
    parser.add_argument("--model", type=str, default="att", help="naive | att")
    parser.add_argument("--glove", action="store_true", help="Use glove as initial word embedding.")
    parser.add_argument("--embedding_size", type=int, default=300,
                        help="Word embedding size. (For glove, use 50 | 100 | 200 | 300)")

    parser.add_argument("--num_hidden", type=int, default=100, help="RNN Network size.")
    parser.add_argument("--num_layers", type=int, default=2, help="RNN Network depth.")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--keep_prob", type=float, default=0.8, help="Dropout keep prob.")
    parser.add_argument("--checkpoint_dir", type=str, default="saved_model", help="Checkpoint directory.")


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()

num_class = 2
if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

print("Building dictionary...")
word_dict, reversed_dict, document_max_len = build_dict(args.train_tsv)
print("Building dataset...")
x, y = build_dataset(args.train_tsv, word_dict, document_max_len)
# Split to train and validation data
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15)


with tf.Session() as sess:
    if args.model == "naive":
        model = NaiveRNN(reversed_dict, document_max_len, num_class, args)
    elif args.model == "att":
        model = AttentionRNN(reversed_dict, document_max_len, num_class, args)
    else:
        raise NotImplementedError()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    train_batches = batch_iter(train_x, train_y, args.batch_size, args.num_epochs)
    num_batches_per_epoch = (len(train_x) - 1) // args.batch_size + 1
    max_accuracy = 0

    for x_batch, y_batch in train_batches:
        train_feed_dict = {
            model.x: x_batch,
            model.y: y_batch,
            model.keep_prob: args.keep_prob
        }

        _, step, loss = sess.run([model.optimizer, model.global_step, model.loss], feed_dict=train_feed_dict)

        if step % 100 == 0:
            print("step {0}: loss = {1}".format(step, loss))

        if step % 2000 == 0:
            # Test accuracy with validation data for each epoch.
            valid_batches = batch_iter(valid_x, valid_y, args.batch_size, 1)
            sum_accuracy, cnt = 0, 0

            for valid_x_batch, valid_y_batch in valid_batches:
                valid_feed_dict = {
                    model.x: valid_x_batch,
                    model.y: valid_y_batch,
                    model.keep_prob: 1.0
                }

                accuracy = sess.run(model.accuracy, feed_dict=valid_feed_dict)
                sum_accuracy += accuracy
                cnt += 1
            valid_accuracy = sum_accuracy / cnt

            print("\nValidation Accuracy = {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))

            # Save model
            if valid_accuracy > max_accuracy:
                max_accuracy = valid_accuracy
                saver.save(sess, "{0}/{1}.ckpt".format(args.checkpoint_dir, args.model), global_step=step)
                print("Model is saved.\n")
