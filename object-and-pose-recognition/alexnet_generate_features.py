import os
import pandas as pd
import tensorflow as tf
from numpy import *
import numpy as np
from scipy.misc import *


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    """From https://github.com/ethereon/caffe-tensorflow
    """
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def init_alex_net(n_data):
    input_dim = (256, 256, 3)
    xdim = (227, 227, 3)

    net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

    input_alex = tf.placeholder(tf.float32, (None,) + input_dim)

    # random cropping
    cropped_input = tf.random_crop(input_alex, (n_data, ) + xdim)

    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11;
    k_w = 11;
    c_o = 96;
    s_h = 4;
    s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(cropped_input, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5;
    k_w = 5;
    c_o = 256;
    s_h = 1;
    s_w = 1;
    group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3;
    k_w = 3;
    c_o = 384;
    s_h = 1;
    s_w = 1;
    group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    # conv4
    # conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3;
    k_w = 3;
    c_o = 384;
    s_h = 1;
    s_w = 1;
    group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    # conv5
    # conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3;
    k_w = 3;
    c_o = 256;
    s_h = 1;
    s_w = 1;
    group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    # maxpool5
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # fc6
    # fc(4096, name='fc6')
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    # fc7
    # fc(4096, name='fc7')
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    return input_alex, fc7


def alex_net_fc7(sess, input_alex, fc7, input_images):
    input_images = [input_image - mean(input_image) for input_image in input_images]
    output_fc7 = sess.run(fc7, feed_dict={input_alex: input_images})

    return output_fc7


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def load_batch(dataframe):
    rgbs = []
    depths = []
    labels = []
    for i in range(dataframe.shape[0]):
        item = dataframe.iloc[i]
        image = imread(item.location).astype(float32)

        rgbs.append(image[:, 0: image.shape[1]//2, :])
        depths.append(image[:, image.shape[1]//2: image.shape[1], :])
        label = item.label[1:-1].split(' ')
        label = [int(x) for x in label]
        labels.append(label)

    return array(rgbs), array(depths), labels


def fuse_batch(dataframe, sess, rgb_data, depth_data):
    # load batch of training samples and convert to float
    rgbs, depths, labels = load_batch(dataframe)

    # get the fused features from AlexNet
    fc7_rgb = alex_net_fc7(sess, rgb_data['input_alex'], rgb_data['fc7'], rgbs)
    fc7_depth = alex_net_fc7(sess, depth_data['input_alex'], depth_data['fc7'], depths)
    # fc_7_fused = fc7_rgb * 0.5 + fc7_depth * 0.5            # add a fusion layer 4096 x 2

    return concatenate([fc7_rgb, fc7_depth], axis=1), labels


def fuse_batch_only_rgb(dataframe, sess, input_alex, fc7):
    # load batch of training samples and convert to float
    rgbs, _, labels = load_batch(dataframe)

    # get the fused features from AlexNet
    fc7_rgb = alex_net_fc7(sess, input_alex, fc7, rgbs)

    return fc7_rgb, labels


def set_up_individual_model(batch_size, n_classes):
    input_alex, fc7 = init_alex_net(batch_size)
    y_ = tf.placeholder(float32, [None, n_classes])

    fc_tuning_classW = weight_variable([4096, n_classes])
    fc_tuning_classb = bias_variable([n_classes])
    fc_tuning_class = tf.nn.xw_plus_b(fc7, fc_tuning_classW, fc_tuning_classb)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc_tuning_class)
    train_step1 = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    # train_step2 = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    train_step2 = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    return {'input_alex': input_alex,
            'fc7': fc7,
            'y_': y_,
            'train_step1': train_step1,
            'train_step2': train_step2,
            'cross_entropy': cross_entropy}


def set_up_network(batch_size, n_sources, n_classes):
    rgb_model_data = set_up_individual_model(batch_size, n_classes)
    depth_model_data = set_up_individual_model(batch_size, n_classes)

    classifier_x = tf.placeholder(tf.float32, shape=[None, 4096*n_sources])
    y_ = tf.placeholder(tf.float32, shape=[None, n_classes])

    fc1_fusW = weight_variable([4096*n_sources, 4096])
    fc1_fusb = bias_variable([4096])
    fc1_fus = tf.nn.relu_layer(classifier_x, fc1_fusW, fc1_fusb)

    fc_classW = weight_variable([4096, n_classes])
    fc_classb = bias_variable([n_classes])
    fc_class = tf.nn.xw_plus_b(fc1_fus, fc_classW, fc_classb)

    return classifier_x, y_, fc_class, rgb_model_data, depth_model_data


def tune_alex_net(rgb_model_data, depth_model_data, train_df, batch_size, saver, sess, model_path):
    input_alex_rgb = rgb_model_data['input_alex']
    input_alex_depth = depth_model_data['input_alex']
    cross_entropy_rgb = rgb_model_data['cross_entropy']
    cross_entropy_depth = depth_model_data['cross_entropy']
    train_step_rgb1 = rgb_model_data['train_step1']
    train_step_rgb2 = rgb_model_data['train_step2']
    train_step_depth1 = depth_model_data['train_step1']
    train_step_depth2 = depth_model_data['train_step2']
    y_rgb = rgb_model_data['y_']
    y_depth = depth_model_data['y_']

    train_df = train_df.sample(frac=1, random_state=2000)

    max_steps = 600
    steps_per_epoch = train_df.shape[0] / batch_size

    current_row = 0

    print "fine-tuning Alexnet"

    for i in range(max_steps):
        # Load the batch metadata
        batch_df = train_df.iloc[current_row: current_row + batch_size] \
            if current_row + batch_size < train_df.shape[0] \
            else train_df.iloc[current_row: train_df.shape[0]] \
            .append(train_df.iloc[0: current_row + batch_size - train_df.shape[0]])

        current_row = current_row + batch_size if current_row + batch_size < train_df.shape[0] else 0

        rgbs, depths, labels = load_batch(batch_df)

        # if i % (steps_per_epoch/4) == 0:
        if i % 100 == 0:
            train_loss_rgb = cross_entropy_rgb.eval(feed_dict={input_alex_rgb: rgbs, y_rgb: labels})
            print 'step %d, rgb train cross entropy = %g' % (i, mean(train_loss_rgb))

            train_loss_depth = cross_entropy_depth.eval(feed_dict={input_alex_depth: depths, y_depth: labels})
            print '========depth train cross entropy = %g' % mean(train_loss_depth)

            saver.save(sess, model_path)

        if i <= 300:
            train_step_rgb1.run(feed_dict={input_alex_rgb: rgbs, y_rgb: labels})
            train_step_depth1.run(feed_dict={input_alex_depth: depths, y_depth: labels})
        else:
            train_step_rgb2.run(feed_dict={input_alex_rgb: rgbs, y_rgb: labels})
            train_step_depth2.run(feed_dict={input_alex_depth: depths, y_depth: labels})


def train_binary_network(train_df, test_df, batch_size, n_epochs, model_path,
                         checkpoint_path='', n_sources=2, is_testing=False):
    saving_path = os.path.join(os.path.split(model_path)[0], "tuning-model")

    if not os.path.isdir(os.path.split(model_path)[0]):
        os.mkdir(os.path.split(model_path)[0])

    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    # train_df = pd.read_csv(train_path).sample(frac=1, random_state=1000)
    # test_df = pd.read_csv(test_path)
    train_df, test_df = lai_et_al_split(train_df, test_df)
    train_df = train_df.sample(frac=1, random_state=1000)

    print 'shuffled training data length = %d' % train_df.shape[0]
    print 'test data length = %d' % test_df.shape[0]

    # max_steps = n_epochs * train_df.shape[0] / batch_size
    max_steps = 20000       # set exactly the same as the paper Eitel et. al
    steps_per_epoch = train_df.shape[0] / batch_size

    n_classes = 51

    classifier_x, y_, fc_class, rgb_model_data, depth_model_data = \
        set_up_network(batch_size, n_sources, n_classes)

    correct_prediction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(fc_class, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc_class)
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    saver = tf.train.Saver(max_to_keep=1)

    tune_alex_net(rgb_model_data, depth_model_data, train_df, batch_size, saver, sess,
                  saving_path + "/tuning")

    if is_testing:
        restore_model(checkpoint_path, sess)
    else:
        current_row = 0
        for i in range(max_steps):
            # Load the batch metadata
            batch_df = train_df.iloc[current_row: current_row + batch_size] \
                if current_row + batch_size < train_df.shape[0] \
                else train_df.iloc[current_row: train_df.shape[0]] \
                .append(train_df.iloc[0: current_row + batch_size - train_df.shape[0]])

            current_row = current_row + batch_size if current_row + batch_size < train_df.shape[0] else 0

            fc_7_fused, labels = fuse_batch(batch_df, sess, rgb_model_data, depth_model_data)
            # fc_7_fused, labels = fuse_batch_only_rgb(batch_df, sess, input_alex, fc7)

            if i % (steps_per_epoch/10) == 0:
                train_accuracy = accuracy.eval(feed_dict={classifier_x: fc_7_fused, y_: labels})
                print 'step %d, training accuracy = %g' % (i, train_accuracy)
                train_loss = cross_entropy.eval(feed_dict={classifier_x: fc_7_fused, y_: labels})
                print '======== train cross entropy = %g' % mean(train_loss)

                tf.summary.scalar("accuracy", accuracy)
                tf.summary.scalar("loss", cross_entropy)
                saver.save(sess, model_path)

            # Train the last 2 layers
            train_step.run(feed_dict={classifier_x: fc_7_fused, y_: labels})

    # Test with the validation set
    n_test_steps = int(ceil(test_df.shape[0] * 1.0 / batch_size))

    accuracies = []
    current_row = 0
    for i in range(n_test_steps):
        # Load the batch metadata
        batch_df = test_df.iloc[current_row: current_row + batch_size] \
            if current_row + batch_size < test_df.shape[0] \
            else test_df.iloc[current_row: test_df.shape[0]] \
            .append(test_df.iloc[0: current_row + batch_size - test_df.shape[0]])

        current_row = current_row + batch_size if current_row + batch_size < test_df.shape[0] else 0

        fc_7_fused, labels = fuse_batch(batch_df, sess, rgb_model_data, depth_model_data)
        # fc_7_fused, labels = fuse_batch_only_rgb(batch_df, sess, input_alex, fc7)

        current_accuracy = accuracy.eval(feed_dict={classifier_x: fc_7_fused, y_: labels})

        if i % 100 == 0:
            print "step %d, %d left, accuracy %g" % (i, n_test_steps - i, current_accuracy)
        accuracies.append(current_accuracy)

    return mean(accuracies)


def lai_et_al_split(train_df, test_df):
    random.seed(1000)

    categories = sort(unique(train_df.category))

    # new_train_df = pd.DataFrame(columns=list(train_df))
    test_sequences = []
    for category in categories:
        category_df = train_df[train_df.category == category]
        instances = unique(category_df.instance_number)
        for instance in instances:
            instance_df = category_df[category_df.instance_number == instance]
            video_nos = unique(instance_df.video_no)
            sequences = []
            for video_no in video_nos:
                video_df = instance_df[instance_df.video_no == video_no]
                instance_df = video_df.sort_values(['instance_number'])
                sequences.append(instance_df.iloc[0: instance_df.shape[0]/3])
                sequences.append(instance_df.iloc[instance_df.shape[0]/3: instance_df.shape[0]*2/3])
                sequences.append(instance_df.iloc[instance_df.shape[0]*2/3: instance_df.shape[0]])

            test_indices = random.choice(9, size=2, replace=False)

            for i in test_indices:
                test_sequences.append(sequences[i])

    for sequence in test_sequences:
        test_df = test_df.append(sequence)
        train_df = train_df.drop(sequence.index)

    return train_df, test_df


def restore_model(path, sess):
    print "restoring models"
    saver = tf.train.import_meta_graph(os.path.join(path, 'fusion-net.meta'))
    checkpoint = tf.train.latest_checkpoint(path)
    saver.restore(sess, checkpoint)


if __name__ == '__main__':
    CHECK_POINT = '/mnt/raid/data/ni/dnn/pduy/eitel-et-al-model/'
    # MODEL_PATH = '/mnt/raid/data/ni/dnn/pduy/eitel-et-al-model/fusion-net'
    MODEL_PATH = CHECK_POINT + 'fusion-net'
    PROCESSED_PAIR_PATH = '/mnt/raid/data/ni/dnn/pduy/eitel-et-al-data/'
    # load_batch(pd.read_csv(os.path.join(PROCESSED_PAIR_PATH, 'train_info.csv')))

    training_data = pd.read_csv(os.path.join(PROCESSED_PAIR_PATH, 'training_set.csv')).sample(frac=1, random_state=1000)
    test_data = pd.read_csv(os.path.join(PROCESSED_PAIR_PATH, 'test_set.csv'))

    for i in range(1):
        accuracy = train_binary_network(training_data,
                                        test_data,
                                        50, 20,
                                        MODEL_PATH,
                                        CHECK_POINT,
                                        is_testing=False)

        try:
            with open(os.path.join(CHECK_POINT, 'temp.txt'), 'r') as f:
                print i + 5
                content = f.read()
        except IOError:
            content = ''

        with open(os.path.join(CHECK_POINT, 'temp.txt'), 'w') as f:
            f.writelines(content + '\n' + 'acc = ' + str(accuracy))

