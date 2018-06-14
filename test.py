import tensorflow as tf
import argparse
from data_utils import build_dict, build_dataset, batch_iter


def add_arguments(parser):
    parser.add_argument("--test_tsv", type=str, default="sample_data/test.tsv", help="Test tsv file.")
    parser.add_argument("--checkpoint_dir", type=str, default="saved_model", help="Checkpoint dir for saved model.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()

print("Loading dictionary...")
word_dict, reversed_dict, document_max_len = build_dict(args.test_tsv, is_train=False)
print("Building test dataset...")
test_x, test_y = build_dataset(args.test_tsv, word_dict, document_max_len)

checkpoint_file = tf.train.latest_checkpoint(args.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        x = graph.get_operation_by_name("x").outputs[0]
        y = graph.get_operation_by_name("y").outputs[0]
        keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        batches = batch_iter(test_x, test_y, args.batch_size, 1)
        sum_accuracy, cnt = 0, 0
        for batch_x, batch_y in batches:
            feed_dict = {
                x: batch_x,
                y: batch_y,
                keep_prob: 1.0
            }

            accuracy_out = sess.run(accuracy, feed_dict=feed_dict)
            sum_accuracy += accuracy_out
            cnt += 1

        print("Test Accuracy : {0}".format(sum_accuracy / cnt))
