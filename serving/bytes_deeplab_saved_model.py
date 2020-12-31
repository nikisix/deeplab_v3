import base64
import json
import numpy as np
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
        DEBUG = False
        if DEBUG:
            ######################################
            # # FROM INSTANCES FILE
            # image_filename = 'tmp_input/owl-toadstuhl.jpg.json'
            # with open(image_filename) as fp:
                # instances = json.load(fp)
            # image_bytes = instances['instances'][0]['image_bytes']['b64']
            # image_bytes = base64.b64decode(image_bytes)
            ######################################
            # FROM image file
            # image_filename = 'tmp_input/owl-toadstuhl.jpg'
            image_filename = 'tmp_input/harpy-eagle.jpg'
            image_string_bytes = tf.read_file(image_filename)
            input_tensor = tf.image.decode_jpeg(image_string_bytes, channels=3)
            input_tensor = tf.expand_dims(input_tensor, 0)
            ######################################


        ## 1 ## Create placeholder for image bitstring
        # image_bytes = tf.placeholder(tf.string, shape=(), name="source")
        # input_tensor = tf.image.decode_image(image_bytes)

        """
        Decode the images. The shape of raw_byte_strings is [batch size]
        (were batch size is determined by how many images are sent), and
        the shape of `input_images` is [batch size, height, width, 3]. It's
        important that all of the images sent have the same dimensions
        or errors will result.
        We have to use a map_fn because decode_raw only works on a single
        image, and we need to decode a batch of images.
        """
        if not DEBUG:
            raw_byte_strings = tf.placeholder(dtype=tf.string, shape=[None], name='source')
            decode = lambda raw_byte_str: tf.image.decode_jpeg(raw_byte_str, channels=3)
            input_tensor = tf.map_fn(decode, raw_byte_strings, dtype=tf.uint8)


        # reshape the input image to its original dimension
        # (_, orig_width, orig_height, _) = sess.run(input_tensor).shape  # materialize shape
        # (_, orig_width, orig_height, _) = tf.shape(input_tensor)
        orig_width = tf.shape(input_tensor)[1]
        orig_height = tf.shape(input_tensor)[2]
        input_tensor = tf.image.resize_images(input_tensor, (513, 513))  # match resnet

        #=================TESTED_END============================================

        #=================UNTESTED==============================================
        # extract the segmentation mask

        # Perform inference on the input image
        logits = network.deeplab_v3(input_tensor, args, is_training=False, reuse=False)

        logits = tf.squeeze(logits)  #TODO delete
        predictions = tf.argmax(logits, axis=2)  # TODO axis=3
        probabilities = tf.nn.softmax(logits)

        #Load Pretrained Weights
        # specify the directory where the pre-trained model weights are stored
        pre_trained_model_dir = os.path.join(FLAGS.work_dir, str(FLAGS.model_id), "train")
        saver = tf.train.Saver()
        # Restore variables from disk.
        saver.restore(sess, os.path.join(pre_trained_model_dir, "model.ckpt"))
        print("Model", str(FLAGS.model_id), "restored.")

        ######## Bounding Box ######## (TODO move to function)

        COW_LABEL_ID = 3  # TODO change to 10
        p1 = tf.where(
                tf.equal(predictions,3),
                predictions,
                tf.to_int64(tf.zeros_like(predictions))
            )


        h_edges = tf.argmax(p1, axis=0)
        v_edges = tf.argmax(p1, axis=1)

        # 5
        upper_ix = h_edges[
                tf.argmin(
                    tf.where(
                            tf.not_equal(h_edges, 0),
                            tf.to_int64(h_edges),
                            tf.to_int64(1000*tf.ones_like(h_edges))
                        )
                    )
                ]

        # 255
        lower_ix = h_edges[
                tf.argmax(
                    tf.where(
                            tf.not_equal(h_edges, 0),
                            tf.to_int64(h_edges),
                            tf.to_int64(tf.zeros_like(h_edges))
                        )
                    )
                ]

        # 79
        left_ix = v_edges[
                tf.argmin(
                    tf.where(
                            tf.not_equal(v_edges, 0),
                            tf.to_int64(v_edges),
                            tf.to_int64(1000*tf.ones_like(v_edges))
                        )
                    )
                ]

        # 465
        right_ix = v_edges[
                tf.argmax(
                    tf.where(
                            tf.not_equal(v_edges, 0),
                            tf.to_int64(v_edges),
                            tf.to_int64(tf.zeros_like(v_edges))
                        )
                    )
                ]
        ####### End Bounding Box #############

        # scale back up to original picture's scale
        upper_ix = tf.cast((upper_ix/513) * tf.cast(orig_height, tf.float64), tf.int32)
        lower_ix = tf.cast((lower_ix/513) * tf.cast(orig_height, tf.float64), tf.int32)
        left_ix  = tf.cast((left_ix/513)  * tf.cast(orig_width , tf.float64), tf.int32)
        right_ix = tf.cast((right_ix/513) * tf.cast(orig_width , tf.float64), tf.int32)

        bbox = tf.concat([[upper_ix, left_ix, lower_ix, right_ix]], axis=0)

        # Create SavedModelBuilder class
        # defines where the model will be exported
        export_path_base = FLAGS.export_model_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # tensor info for signature
        tensor_info_input = tf.saved_model.utils.build_tensor_info(raw_byte_strings)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(bbox)

        # Defines the DeepLab signatures, uses the TF Predict API
        # It receives an image and its dimensions and output the segmentation mask
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                # must end with "_bytes"! (see link_1 above)
                inputs={'image_bytes': tensor_info_input},
                outputs={'bounding_box': tensor_info_output},
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
