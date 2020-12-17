import base64
import json
import os
import sys
sys.path.append('../')

import network
import tensorflow as tf


# TODO ON DEPLOY uncomment and set the GPU id if applicable.
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

tf.app.flags.DEFINE_integer('model_version', 1, 'Models version number.')
tf.app.flags.DEFINE_string('work_dir', './tboard_logs', 'Working directory.')
tf.app.flags.DEFINE_integer('model_id', 16645, 'Model id name to be loaded.')
tf.app.flags.DEFINE_string('export_model_dir', "./versions", 'Directory where the model exported files should be placed.')

FLAGS = tf.app.flags.FLAGS

with open('./tboard_logs/16645/train/data.json', 'r') as fp:
    args = json.load(fp)


# input_bytes = ''
DEBUG = False
if DEBUG:
    image_filename = 'tmp_input/owl-toadstuhl.jpg'
    with open(image_filename) as fp:
        instances = json.load(fp)
    image_bytes = instances['instances'][0]['image_bytes']['b64']
    image_bytes = base64.b64decode(image_bytes)

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = Dotdict(args)

# link_1
# https://cloud.google.com/ai-platform/prediction/docs/online-predict#binary_data_in_prediction_input
def main(_):
    with tf.Session(graph=tf.Graph()) as sess:
        # image_string = tf.read_file(image_filename)

        ## 1 ## Create placeholder for image bitstring
        # image_bytes = tf.placeholder(tf.string, shape=(), name="source")
        # input_tensor = tf.image.decode_image(image_bytes)

        """ ERROR 1
        Cannot feed value of shape (1,) for Tensor \'image_bytes:
        0\', which has shape \'()\' (Error code: 2)
        """

        """ ERROR 2
        int() argument must be a string, a bytes-like object or a number, not \'dict\' (Error code: 2)
        """

        """
        Decode the images. The shape of raw_byte_strings is [batch size]
        (were batch size is determined by how many images are sent), and
        the shape of `input_images` is [batch size, 320, 240, 3]. It's
        important that all of the images sent have the same dimensions
        or errors will result.
        We have to use a map_fn because decode_raw only works on a single
        image, and we need to decode a batch of images.
        """
        raw_byte_strings = tf.placeholder(dtype=tf.string, shape=[None], name='source')
        # decode = lambda raw_byte_str: tf.decode_raw(raw_byte_str, tf.uint8)
        decode = lambda raw_byte_str: tf.image.decode_jpeg(raw_byte_str, channels=3)
        input_tensor = tf.map_fn(decode, raw_byte_strings, dtype=tf.uint8)

        # TODO include?
        # input_tensor = tf.to_float(tf.image.convert_image_dtype(input_tensor, dtype=tf.float32))
        # input_tensor = input_tensor / 127.5 - 1.0

        # add batch dimension
        input_tensor.set_shape([None, None, None, 3])

        # reshape the input image to its original dimension
        # TODO store the original size of image
        input_tensor = tf.image.resize_images(input_tensor, (513, 513))  # match resnet

        # perform inference on the input image
        logits_tf = network.deeplab_v3(input_tensor, args, is_training=False, reuse=False)

        #=================TESTED_END============================================

        #=================UNTESTED==============================================
        # extract the segmentation mask
        predictions_tf = tf.argmax(logits_tf, axis=3)

        # specify the directory where the pre-trained model weights are stored
        pre_trained_model_dir = os.path.join(FLAGS.work_dir, str(FLAGS.model_id), "train")

        saver = tf.train.Saver()

        # Restore variables from disk.
        saver.restore(sess, os.path.join(pre_trained_model_dir, "model.ckpt"))
        print("Model", str(FLAGS.model_id), "restored.")

        # Create SavedModelBuilder class
        # defines where the model will be exported
        export_path_base = FLAGS.export_model_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)


        # tensor info for signature
        # tensor_info_input = tf.saved_model.utils.build_tensor_info(image_bytes)
        tensor_info_input = tf.saved_model.utils.build_tensor_info(raw_byte_strings)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(predictions_tf)

        # Defines the DeepLab signatures, uses the TF Predict API
        # It receives an image and its dimensions and output the segmentation mask
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                # must end with "_bytes"! (see link_1 above)
                inputs={'image_bytes': tensor_info_input},
                outputs={'segmentation_map': tensor_info_output},
                method_name=tf.saved_model.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.SERVING],
            signature_def_map={
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: #'serving_default':
                    prediction_signature,
            })

        # export the model
        builder.save(as_text=True)
        print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()
