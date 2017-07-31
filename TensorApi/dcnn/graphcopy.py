import argparse
import sys

import inception as inception
import tensorflow as tf


FLAGS = None

def main(_):
    # Load the Inception model so it is ready for classifying images.
    model = inception.Inception()

    graph = model.graph



    # Use the Inception model to classify the image.
    pred = model.classify(image_path='C:\\Users\\emman\\PycharmProjects\\TensorWebApi\\models\\inception\\test7.jpg')

    # Print the scores and names for the top-10 predictions.
    #model.print_scores(pred=pred, k=10)

    # Write summary for TensorBoard
    model._write_summary(logdir='/tmp/tensorflow_logs/example')

    # Close the TensorFlow session.
    model.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)