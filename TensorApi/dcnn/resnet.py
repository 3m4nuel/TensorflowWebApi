import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
from tensorflow.models.slim.nets import inception_resnet_v2
from tensorflow.models.slim.preprocessing import inception_preprocessing
import numpy as np

class Resnet:
    def __init__(self):
        self.session = tf.Session()
        self.graph = self.session.graph

    def _write_summary(self, logdir='summary/'):
        """
        Write graph to summary-file so it can be shown in TensorBoard.
        This function is used for debugging and may be changed or removed in the future.
        :param logdir:
            Directory for writing the summary-files.
        :return:
            Nothing.
        """

        writer = tf.summary.FileWriter(logdir=logdir, graph=self.graph)
        writer.close()

    def classify(self, image_path=None):
        #Load the model
        input_tensor = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='input_image')
        scaled_input_tensor = tf.scalar_mul((1.0 / 255), input_tensor)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

        arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_resnet_v2(scaled_input_tensor, is_training=False)

        variables_to_restore = slim.get_model_variables()
        print(variables_to_restore)

        saver = tf.train.Saver()
        saver.restore(self.session, "C:\\Users\\emman\\PycharmProjects\\TensorWebApi\\models\\inception\\inception_resnet_v2_2016_08_30.ckpt")

        image = Image.open(image_path).resize((299, 299))
        processed_image = inception_preprocessing.preprocess_image(image,
                                                                    inception_resnet_v2.default_image_size,
                                                                    inception_resnet_v2.default_image_size,
                                                                    is_training=False)


        predict_values, logit_values = self.session.run([end_points['Predictions'], logits], feed_dict={input_tensor: processed_image})
        print(np.max(predict_values), np.max(logit_values))
        print(np.argmax(predict_values), np.argmax(logit_values))

        return predict_values