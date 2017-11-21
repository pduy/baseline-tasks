import os
from os.path import *
import pandas as pd
import tensorflow as tf
import numpy as np
from scipy.misc import *
import time


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

    net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

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
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    # fc7
    # fc(4096, name='fc7')
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    return input_alex, fc7


def alex_net_fc7(sess, input_alex, fc7, input_images):
    input_images = [input_image - np.mean(input_image) for input_image in input_images]
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
        image = imread(item.location).astype(np.float32)

        rgbs.append(image[:, 0: image.shape[1]//2, :])
        depths.append(image[:, image.shape[1]//2: image.shape[1], :])
        label = item.label[1:-1].split(' ')
        label = [int(x) for x in label]
        labels.append(label)

    return np.array(rgbs), np.array(depths), labels


def fuse_batch(dataframe, sess, rgb_data, depth_data):
    # load batch of training samples and convert to float
    rgbs, depths, labels = load_batch(dataframe)

    # get the fused features from AlexNet
    fc7_rgb = alex_net_fc7(sess, rgb_data['input_alex'], rgb_data['fc7'], rgbs)
    fc7_depth = alex_net_fc7(sess, depth_data['input_alex'], depth_data['fc7'], depths)
    # fc_7_fused = fc7_rgb * 0.5 + fc7_depth * 0.5            # add a fusion layer 4096 x 2

    return np.concatenate([fc7_rgb, fc7_depth], axis=1), labels


def fuse_batch_only_rgb(dataframe, sess, input_alex, fc7):
    # load batch of training samples and convert to float
    rgbs, _, labels = load_batch(dataframe)

    # get the fused features from AlexNet
    fc7_rgb = alex_net_fc7(sess, input_alex, fc7, rgbs)

    return fc7_rgb, labels


def load_representations(data_frame, use_depth=True):
    reps = []
    labels = []
    for i in range(data_frame.shape[0]):
        current_row = data_frame.iloc[i]
        rgb_file = current_row.rgb_rep_location
        depth_file = current_row.rgb_generated_rep_location

        label = current_row.label[1:-1].split(' ')
        label = [int(x) for x in label]
        labels.append(label)

        rgb_vector = np.loadtxt(rgb_file)

        if use_depth:
            depth_vector = np.loadtxt(depth_file)
            rep = np.concatenate([rgb_vector, depth_vector], axis=0)
            reps.append(rep)
        else:
            reps.append(rgb_vector)

    return np.array(reps), np.array(labels)


def set_up_individual_model(batch_size, n_classes):
    input_alex, fc7 = init_alex_net(batch_size)
    y_ = tf.placeholder(np.float32, [None, n_classes])

    fc_tuning_classW = weight_variable([4096, n_classes])
    fc_tuning_classb = bias_variable([n_classes])
    fc_tuning_class = tf.nn.xw_plus_b(fc7, fc_tuning_classW, fc_tuning_classb)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc_tuning_class)
    train_step1 = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cross_entropy)
    train_step2 = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cross_entropy)

    return {'input_alex': input_alex,
            'fc7': fc7,
            'y_': y_,
            'train_step1': train_step1,
            'train_step2': train_step2,
            'cross_entropy': cross_entropy}


def set_up_network(batch_size, n_sources, n_classes, need_individual_network=False):
    classifier_x = tf.placeholder(tf.float32, shape=[None, 4096*n_sources])
    y_ = tf.placeholder(tf.float32, shape=[None, n_classes])

    fc1_fusW = weight_variable([4096*n_sources, 4096])
    fc1_fusb = bias_variable([4096])
    fc1_fus = tf.nn.relu_layer(classifier_x, fc1_fusW, fc1_fusb)

    fc_classW = weight_variable([4096, n_classes])
    fc_classb = bias_variable([n_classes])
    fc_class = tf.nn.xw_plus_b(fc1_fus, fc_classW, fc_classb)

    if need_individual_network:
        rgb_model_data = set_up_individual_model(batch_size, n_classes)
        depth_model_data = set_up_individual_model(batch_size, n_classes)

        return classifier_x, y_, fc_class, rgb_model_data, depth_model_data
    else:
        return classifier_x, y_, fc_class, fc1_fusW


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

    max_steps = 10
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
            print 'step %d, rgb train cross entropy = %g' % (i, np.mean(train_loss_rgb))

            train_loss_depth = cross_entropy_depth.eval(feed_dict={input_alex_depth: depths, y_depth: labels})
            print '========depth train cross entropy = %g' % np.mean(train_loss_depth)

            saver.save(sess, model_path)

        if i <= 300:
            train_step_rgb1.run(feed_dict={input_alex_rgb: rgbs, y_rgb: labels})
            train_step_depth1.run(feed_dict={input_alex_depth: depths, y_depth: labels})
        else:
            train_step_rgb2.run(feed_dict={input_alex_rgb: rgbs, y_rgb: labels})
            train_step_depth2.run(feed_dict={input_alex_depth: depths, y_depth: labels})


def train_binary_network(train_df, test_df, batch_size, n_epochs,
                         checkpoint_path='', n_sources=2, is_testing=False):

    n_classes = 51

    classifier_x, y_, fc_class, fc1_fusW = \
        set_up_network(batch_size, n_sources, n_classes)

    correct_prediction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(fc_class, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc_class)
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    saver = tf.train.Saver(max_to_keep=1)

    # tune_alex_net(rgb_model_data, depth_model_data, train_df, batch_size, saver, sess,
    #               saving_path + "/tuning")

    if is_testing:
        restore_model(checkpoint_path, sess)

        print 'batch size = %d' % batch_size

        categories = test_df.category.unique()

        all_accuracies=[]
        for category in categories:
            sub_test_df = test_df[test_df.category == category]
            print 'testing category %s' % category
            print 'number of test items = %d' % sub_test_df.shape[0]

            # Test with the validation set
            n_test_steps = int(np.ceil(sub_test_df.shape[0] * 1.0 / batch_size))

            accuracies = []
            current_row = 0
            for test_step in range(n_test_steps):
                # Load the batch metadata
                batch_df = sub_test_df.iloc[current_row: current_row + batch_size] \
                    if current_row + batch_size < sub_test_df.shape[0] \
                    else sub_test_df.iloc[current_row: sub_test_df.shape[0]]
                # .append(sub_test_df.iloc[0: current_row + batch_size - sub_test_df.shape[0]])

                current_row = current_row + batch_size if current_row + batch_size < sub_test_df.shape[0] else 0

                if n_sources == 2:
                    fc_7_fused, labels = load_representations(batch_df)
                else:
                    fc_7_fused, labels = load_representations(batch_df, use_depth=False)
                # fc_7_fused, labels = fuse_batch_only_rgb(batch_df, sess, input_alex, fc7)

                current_accuracy = accuracy.eval(feed_dict={classifier_x: fc_7_fused, y_: labels})

                # if test_step % 500 == 0:
                #     print "step %d, %d left, accuracy %g" % (test_step, n_test_steps - test_step, current_accuracy)
                accuracies.append(current_accuracy)

            current_mean_accuracy = np.mean(accuracies)
            all_accuracies.append({'category': category, 'accuracy': current_mean_accuracy})
            print "mean accuracy of '{0}' = '{1}'".format(category, current_mean_accuracy)

        all_accuracies = pd.DataFrame(all_accuracies)
        return all_accuracies
    else:
        model_path = join(checkpoint_path, 'fusion-net')
        if not isdir(split(model_path)[0]):
            os.makedirs(split(model_path)[0])

        max_steps = n_epochs * train_df.shape[0] / batch_size
        # max_steps = 20000  # set exactly the same as the paper Eitel et. al
        steps_per_epoch = train_df.shape[0] / batch_size

        print 'shuffled training data length = %d' % train_df.shape[0]
        print 'test data length = %d' % test_df.shape[0]

        writer = tf.summary.FileWriter(split(model_path)[0], sess.graph)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("loss", tf.reduce_mean(cross_entropy))
        summary_op = tf.summary.merge_all()

        current_row = 0
        for cur_train_step in range(max_steps):
            # Load the batch metadata
            batch_df = train_df.iloc[current_row: current_row + batch_size] \
                if current_row + batch_size < train_df.shape[0] \
                else train_df.iloc[current_row: train_df.shape[0]] \
                .append(train_df.iloc[0: current_row + batch_size - train_df.shape[0]])

            current_row = current_row + batch_size if current_row + batch_size < train_df.shape[0] else 0

            # fc_7_fused, labels = fuse_batch(batch_df, sess, rgb_model_data, depth_model_data)
            # fc_7_fused, labels = fuse_batch_only_rgb(batch_df, sess, input_alex, fc7)
            if n_sources == 2:
                fc_7_fused, labels = load_representations(batch_df)
            else:
                fc_7_fused, labels = load_representations(batch_df, use_depth=False)

            if cur_train_step % (steps_per_epoch/10) == 0:
                train_accuracy = accuracy.eval(feed_dict={classifier_x: fc_7_fused, y_: labels})
                print 'step %d, training accuracy = %g' % (cur_train_step, train_accuracy)
                train_loss = cross_entropy.eval(feed_dict={classifier_x: fc_7_fused, y_: labels})
                print '======== train cross entropy = %g' % np.mean(train_loss)

                saver.save(sess, model_path)

            # Train the last 2 layers
            # train_step.run(feed_dict={classifier_x: fc_7_fused, y_: labels})

            _, summary = sess.run(fetches=[train_step, summary_op], feed_dict={classifier_x: fc_7_fused, y_: labels})
            writer.add_summary(summary,
                               global_step=cur_train_step)
        return 0


def lai_et_al_split(train_df, test_df, n_sampling_step=1, seed=1000):
    np.random.seed(seed)
    categories = np.sort(np.unique(train_df.category))

    # new_train_df = pd.DataFrame(columns=list(train_df))
    test_sequences = []
    for category in categories:
        category_df = train_df[train_df.category == category]
        instances = np.unique(category_df.instance_number)
        for instance in instances:
            instance_df = category_df[category_df.instance_number == instance]
            video_nos = np.unique(instance_df.video_no)
            sequences = []
            for video_no in video_nos:
                video_df = instance_df[instance_df.video_no == video_no]
                instance_df = video_df.sort_values(['instance_number'])
                sequences.append(instance_df.iloc[0: instance_df.shape[0]/3])
                sequences.append(instance_df.iloc[instance_df.shape[0]/3: instance_df.shape[0]*2/3])
                sequences.append(instance_df.iloc[instance_df.shape[0]*2/3: instance_df.shape[0]])

            # do this in order to pick the n_th train-test split in the 10-cv setting
            for _ in range(n_sampling_step - 1):
                np.random.choice(9, size=2, replace=False)
            test_indices = np.random.choice(9, size=2, replace=False)

            for i in test_indices:
                test_sequences.append(sequences[i])

    for sequence in test_sequences:
        test_df = test_df.append(sequence)
        train_df = train_df.drop(sequence.index)

    return train_df, test_df


def restore_model(path, sess):
    print "restoring models"
    saver = tf.train.import_meta_graph(join(path, 'fusion-net.meta'))
    checkpoint = tf.train.latest_checkpoint(path)
    saver.restore(sess, checkpoint)
    return sess


# def create_washington_representations(data_frame, saving_path, generated_portion=0.0):
def create_washington_representations(data_frame, saving_path, is_testing=True):
    if isfile(saving_path):
        return pd.read_csv(saving_path)

    if not isdir(split(saving_path)[0]):
        os.makedirs(split(saving_path)[0])

    input_alex, fc7 = init_alex_net(1)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    rep_data = []
    for item_index in range(data_frame.shape[0]):
        current_item = data_frame.iloc[item_index]

        try:
            location = current_item.location_generated
            combined_image = imread(location)
        except (IOError, AttributeError):
            if is_testing:
                location = current_item.location
                combined_image = imread(location)
            else:
                continue

        print 'processing ' + location
        rgb_image = combined_image[:, 0: combined_image.shape[1]//2, :]
        rgb_image_generated = combined_image[:, combined_image.shape[1]//2: combined_image.shape[1], :]
        # depth_image = combined_image[:, combined_image.shape[1]//2: combined_image.shape[1], :]

        rgb_fc7 = alex_net_fc7(sess, input_alex, fc7, [rgb_image])
        rgb_fc7_generated = alex_net_fc7(sess, input_alex, fc7, [rgb_image_generated])
        # depth_fc7 = alex_net_fc7(sess, input_alex, fc7, [depth_image])

        rgb_file = splitext(split(location)[1])[0] + 'rgb.dat'
        rgb_file = join(split(saving_path)[0], rgb_file)
        rgb_file_generated = splitext(split(location)[1])[0] + 'rgb_generated.dat'
        rgb_file_generated = join(split(saving_path)[0], rgb_file_generated)
        # depth_file = splitext(split(location)[1])[0] + 'depth.dat'
        # depth_file = join(split(saving_path)[0], depth_file)

        np.savetxt(rgb_file, rgb_fc7)
        np.savetxt(rgb_file_generated, rgb_fc7_generated)

        rep_data.append({
            'rgb_rep_location': rgb_file,
            'rgb_generated_rep_location': rgb_file_generated,
            'label': current_item.label,
            'category': current_item.category,
            'instance_number': current_item.instance_number,
            'video_no': current_item.video_no,
            'frame_no': current_item.frame_no
        })

    df = pd.DataFrame(rep_data)
    df.to_csv(saving_path, index=False)

    return df

def get_incomplete_gan_training_data(rep_df, portion):
    gan_list = os.listdir('/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-50-test/processed-images')
    marks = []
    for i in range(rep_df.shape[0]):
        current_item = rep_df.iloc[i]
        name = '_'.join([current_item.category, str(int(current_item.instance_number)), 
                        str(int(current_item.video_no)), str(int(current_item.frame_no)), 'crop-inputs']) \
                    + '.png'
        if name in gan_list:
            print 'file %s exists in GAN data' % name
            current_mark = 1
        else:
            #print 'file %s doesn''t exist' % name
            current_mark = 0

        marks.append(current_mark)

    rep_df['origin'] = pd.Series(marks)

    original_data = rep_df[rep_df.origin == 0]
    gan_data = rep_df[rep_df.origin == 1]
    gan_data = gan_data.sample(frac=portion, random_state=2000)

    return original_data.append(gan_data).sample(frac=1, random_state=1000)


def train_or_test_model_from_csv(train_df, test_df, split_index, data_fraction, checkpoint_to_save, is_testing=True):
    training_split_lai, test_split_lai = lai_et_al_split(train_df, test_df, n_sampling_step=split_index)

    sampled_training = training_split_lai.sample(frac=data_fraction, random_state=1000)

    g = tf.Graph()
    with g.as_default():
        test_accuracies = train_binary_network(sampled_training,
                             test_split_lai,
                             200, 8,
                             join(checkpoint_to_save, 'iter_' + str(split_index)),
                             n_sources=2,
                             is_testing=is_testing)

        if is_testing:
            print 'acc = ' + str(np.mean(test_accuracies.accuracy))
            with open(os.path.join(checkpoint_to_save, 'temp_8_epochs.txt'), 'a+') as f:
                f.writelines('acc = ' + str(np.mean(test_accuracies.accuracy)) + '\n')
            test_accuracies.to_csv(join(checkpoint_to_save, 'iter_' + str(split_index), 'category-accuracies.csv'),
                                   index=False)

    g = None


if __name__ == '__main__':
    CHECK_POINT_BASE = '/mnt/raid/data/ni/dnn/pduy/eitel-et-al-model/'

    CHECK_POINT_100 = join(CHECK_POINT_BASE, '100-0')
    CHECK_POINT_50 = join(CHECK_POINT_BASE, '50-0')
    CHECK_POINT_25 = join(CHECK_POINT_BASE, '25-0')
    CHECK_POINT_10 = join(CHECK_POINT_BASE, '10-0')
    CHECK_POINT_10_90 = join(CHECK_POINT_BASE, '10-90')
    CHECK_POINT_25_75 = join(CHECK_POINT_BASE, '25-75')
    CHECK_POINT_50_50 = join(CHECK_POINT_BASE, '50-50')
    CHECK_POINT_50_20 = join(CHECK_POINT_BASE, '50-20')
    CHECK_POINT_50_30 = join(CHECK_POINT_BASE, '50-30')
    CHECK_POINT_50_40 = join(CHECK_POINT_BASE, '50-40')

    CHECK_POINT_100_RGB_ONLY = join(CHECK_POINT_BASE, '100-0-rgb')
    CHECK_POINT_50_RGB_ONLY = join(CHECK_POINT_BASE, '50-0-rgb')
    CHECK_POINT_25_RGB_ONLY = join(CHECK_POINT_BASE, '25-0-rgb')
    CHECK_POINT_10_RGB_ONLY = join(CHECK_POINT_BASE, '10-0-rgb')

    PROCESSED_PAIR_PATH = '/mnt/raid/data/ni/dnn/pduy/eitel-et-al-data/'
    REPRESENTATION_PATH_TRAINING = '/mnt/raid/data/ni/dnn/pduy/alex_rep/training/alex_rep_training.csv'
    REPRESENTATION_PATH_TEST = '/mnt/raid/data/ni/dnn/pduy/alex_rep/test/alex_rep_test.csv'
    REPRESENTATION_PATH_GAN_TRAIN_50 = '/mnt/raid/data/ni/dnn/pduy/alex_rep/gan_train_50/alex_rep_gan_train_50.csv'
    REPRESENTATION_PATH_GAN_TRAIN_25 = '/mnt/raid/data/ni/dnn/pduy/alex_rep/gan_train_25/alex_rep_gan_train_25.csv'
    REPRESENTATION_PATH_GAN_TRAIN_10 = '/mnt/raid/data/ni/dnn/pduy/alex_rep/gan_train_10/alex_rep_gan_train_10.csv'

    CSV_AGGREGATED_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset-interpolated-aggregated.csv'
    GAN_PROCESSED_CSV_50 = '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-50-test/processed-images' \
                           '/gan-test-data.csv'
    GAN_PROCESSED_CSV_25 = '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-25-test/processed-images' \
                           '/gan-test-data.csv'
    GAN_PROCESSED_CSV_10 = '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-10-test/processed-images' \
                           '/gan-test-data.csv'


    '''Stereo RGB classifier'''
    RGB_STEREO_TRAIN_CSV = '/mnt/raid/data/ni/dnn/pduy/eitel-et-al-data-stereo-rgb/training_set.csv'
    RGB_STEREO_TEST_CSV = '/mnt/raid/data/ni/dnn/pduy/eitel-et-al-data-stereo-rgb/test_set.csv'
    # GAN_PROCESSED_CSV_STEREO_RGB = '/mnt/raid/data/ni/dnn/pduy/training-pose-16bit/' \
    #                             'rgbd-50-reg-discrim-instance-noise-smooth-label-filtering-categories-test/' \
    #                             'processed-images-stereo-rgb/gan-test-data.csv'
    REPRESENTATION_PATH_TRAIN_STEREO_RGB = '/mnt/raid/data/ni/dnn/pduy/alex_rep/train_stereo_rgb/' \
                                               'alex_rep_train_stereo_rgb.csv'
    # REPRESENTATION_PATH_GAN_TRAIN_STEREO_RGB = '/mnt/raid/data/ni/dnn/pduy/alex_rep/gan_train_stereo_rgb_50/' \
    #                                            'alex_rep_gan_train_stereo_rgb.csv'
    REPRESENTATION_PATH_TEST_STEREO_RGB = '/mnt/raid/data/ni/dnn/pduy/alex_rep/test_stereo_rgb/alex_rep_test_stereo_rgb.csv'
    # CHECK_POINT_STEREO_RGB_50 = join(CHECK_POINT_BASE, 'stereo_rgb_50')
    CHECK_POINT_STEREO_RGB_ORIGINAL = join(CHECK_POINT_BASE, 'stereo_rgb_original')

    training_data = pd.read_csv(RGB_STEREO_TRAIN_CSV).sample(frac=1, random_state=1000)
    test_data = pd.read_csv(RGB_STEREO_TEST_CSV)

    training_rep_data = create_washington_representations(training_data,
                                                          REPRESENTATION_PATH_TRAIN_STEREO_RGB,
                                                          is_testing=True)

    test_rep_data = create_washington_representations(test_data, REPRESENTATION_PATH_TEST_STEREO_RGB)

    for i in range(1, 4):
        train_or_test_model_from_csv(train_df=training_rep_data, test_df=test_rep_data, split_index=i,
                                     data_fraction=1, checkpoint_to_save=CHECK_POINT_STEREO_RGB_ORIGINAL, is_testing=False)
        train_or_test_model_from_csv(train_df=training_rep_data, test_df=test_rep_data, split_index=i,
                                     data_fraction=1, checkpoint_to_save=CHECK_POINT_STEREO_RGB_ORIGINAL, is_testing=True)
