import time as time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.copy_graph as cg
import tensorflow.models.slim.datasets.imagenet as imagenet
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.models.slim.preprocessing import inception_preprocessing


########################################################################

class Inception:
    def __init__(self):
        self.session = tf.Session()
        self.graph = self.session.graph

    def close(self):
        """
        Call this function when you are done using the Inception model.
        It closes the TensorFlow session to release its resources.
        """
        self.session.close()

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
        image = tf.gfile.FastGFile(image_path, 'rb').read()
        image_data = tf.image.decode_jpeg(image, channels=3)
        processed_image = inception_preprocessing.preprocess_image(image_data,
                                                                  inception.inception_v3.default_image_size,
                                                                  inception.inception_v3.default_image_size,
                                                                  is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(processed_images, num_classes=1001, is_training=False)

        # In order to get probabilities we apply softmax on the output.
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            "C:\\Users\\emman\\PycharmProjects\\TensorWebApi\\models\\inception\\inception_v3.ckpt",
            slim.get_model_variables('InceptionV3'))

        init_fn(self.session)

        list = []
        for op in self.graph.get_operations():
            for i in ["Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e",
                 "Mixed_7a", "Mixed_7b", "Mixed_7c"]:
                if i in str(op.name):
                    list.append(op)
                    break

        to_graph = tf.Graph()

        for i in list:
            cg.copy_op_to_graph(org_instance=i, to_graph=to_graph, variables="")

        tf.reset_default_graph()

        self.graph = to_graph

        #print(list)
       ## for a in self.graph.get_operations():
           ## print(str(a.name))

        start_time = time.time()

        np_image, network_input, probabilities = self.session.run([image_data,
                                                           processed_image,
                                                           probabilities])

        duration = time.time() - start_time
        print("Compute time: " + str(duration))

        probabilities = probabilities[0, 0:]


        return probabilities

    def print_scores(self, probabilities):
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]
        names = imagenet.create_readable_names_for_imagenet_labels()
        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.4f => [%s]' % (probabilities[index], names[index + 1]))

    def create_graph(self):
        list = []
        for op in self.graph.get_operations():
            if 'Mixed' in str(op.name):
                break
            else:
                list.append(op)

        to_graph = tf.Graph()

        for i in list:
            cg.copy_op_to_graph(org_instance=i, to_graph=to_graph, variables="")

        self.graph = to_graph

        self.session.close()

        #self.session = tf.Session()

        saver = tf.train.Saver()
        tf.reset_default_graph()
       # self.session.run(tf.global_variables_initializer())
        saver.export_meta_graph("C:\\tmp\\model-meta\\raspi-model.meta")

        print("hello")
        for a in self.graph.get_operations():
            print(str(a.name))



########################################################################