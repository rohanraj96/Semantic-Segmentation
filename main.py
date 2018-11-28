import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
    
    layer8 = tf.layers.conv2d(layer7, filters = num_classes, kernel_size=1, strides = 1, 
                                  padding='same', kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    layer9 = tf.layers.conv2d_transpose(layer8, filters = layer4.get_shape().as_list()[-1], kernel_size = 4, strides = (2,2), padding='same',
                                  kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    layer9_skip_connected = tf.add(layer9, layer4)
    
    layer10 = tf.layers.conv2d_transpose(layer9_skip_connected, filters=layer3.get_shape().as_list()[-1], kernel_size=4, strides=(2, 2), 
                                         padding='SAME',kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                         kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    layer10_skip_connected = tf.add(layer10, layer3)
    
    layer11 = tf.layers.conv2d_transpose(layer10_skip_connected, filters=num_classes, kernel_size=16, strides=(8, 8), padding='SAME',
                                         kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                         kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    return layer11

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

    sess.run(tf.global_variables_initializer())
    
    print("Training started ...")
    print()
    
    for i in range(epochs):
        print("EPOCH {} - ".format(i+1))
        
        for image, label in get_batches_fn(batch_size):
            _, loss= sess.run([train_op, cross_entropy_loss], feed_dict  = {input_image:image, correct_label:label, 
                                                                            keep_prob:0.5, learning_rate:0.0009})
            print("LOSS = {:.3f}".format(loss))
        print()
tests.test_train_nn(train_nn)


def run():

    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:

        vgg_path = os.path.join(data_dir, 'vgg')
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)
        epochs = 50
        batch_size = 5
        
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        input_image , keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        final_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(final_layer, correct_label, learning_rate, num_classes)
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
