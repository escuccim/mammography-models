import numpy as np
import os
import wget
from sklearn.model_selection import train_test_split
import tensorflow as tf
from training_utils import download_file, get_batches, read_and_decode_single_example, load_validation_data, \
    download_data, evaluate_model, get_training_data
import sys
import argparse
from tensorboard import summary as summary_lib

# download the data
download_data(what="newest")
# ## Create Model

## config
# If number of epochs has been passed in use that, otherwise default to 50
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epochs to train", type=int)
args = parser.parse_args()

if args.epochs:
    epochs = args.epochs
else:
    epochs = 50

batch_size = 64

train_files, total_records = get_training_data(type="newest")

## Hyperparameters
# Small epsilon value for the BN transform
epsilon = 1e-8

# learning rate
epochs_per_decay = 5
starting_rate = 0.001
decay_factor = 0.85
staircase = True

# learning rate decay variables
steps_per_epoch = int(total_records / batch_size)
print("Steps per epoch:", steps_per_epoch)

# lambdas
lamC = 0.00010
lamF = 0.00200

# use dropout
dropout = True
fcdropout_rate = 0.5
convdropout_rate = 0.01
pooldropout_rate = 0.2

num_classes = 2

## Build the graph
graph = tf.Graph()

# whether to retrain model from scratch or use saved model
init = True
model_name = "model_s1.0.4.01b"
# 0.0.4.01 - starting from scratch

def _conv2d_batch_norm(input, filters, kernel_size=(3,3), stride=(1,1), padding="SAME", seed=None, lambd=0.0, name=None):
    conv = tf.layers.conv2d(
        input,  # Input data
        filters=filters,  # 32 filters
        kernel_size=kernel_size,  # Kernel size: 5x5
        strides=stride,  # Stride: 2
        padding=padding,  # "same" padding
        activation=None,  # None
        kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=seed),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambd),
        name='conv_'+name
    )

    conv = tf.layers.batch_normalization(
        conv,
        axis=-1,
        momentum=0.99,
        epsilon=epsilon,
        center=True,
        scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        training=training,
        name='bn_'+name
    )

    # apply relu
    conv = tf.nn.relu(conv, name='relu_'+name)

    return conv

with graph.as_default():
    training = tf.placeholder(dtype=tf.bool, name="is_training")

    # create global step for decaying learning rate
    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(starting_rate,
                                               global_step,
                                               steps_per_epoch * epochs_per_decay,
                                               decay_factor,
                                               staircase=staircase)

    with tf.name_scope('inputs') as scope:
        image, label = read_and_decode_single_example(train_files, label_type="label_normal", normalize=False)

        X_def, y_def = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=2000,
                                              min_after_dequeue=1000)

        # Placeholders
        X = tf.placeholder_with_default(X_def, shape=[None, 299, 299, 1])
        y = tf.placeholder_with_default(y_def, shape=[None])

        X = tf.cast(X, dtype=tf.float32)

    # Convolutional layer 1
    with tf.name_scope('conv1') as scope:
        conv1 = _conv2d_batch_norm(X, filters=32, padding="VALID", name="1.0")

    # reduce dimensionality
    with tf.name_scope('pool1') as scope:
        pool1 = tf.layers.average_pooling2d(
            conv1,  # Input
            pool_size=(3, 3),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='VALID',  # "same" padding
            name='pool1'
        )

    # conv layer 2 branch 1
    with tf.name_scope('conv2.1') as scope:
        conv21 = _conv2d_batch_norm(pool1, filters=64, padding="SAME", name="2.1")

    with tf.name_scope('conv2.2') as scope:
        conv21 = _conv2d_batch_norm(conv21, filters=64, padding="SAME", name="2.2")

    # conv layer 2 branch 2
    with tf.name_scope('conv2.3') as scope:
        conv22 = _conv2d_batch_norm(pool1, filters=64, padding="SAME", name="2.3")

    # concat the results
    with tf.name_scope("concat2") as scope:
        concat1 = tf.concat(
            [conv21, conv22],
            axis=3,
            name='concat2'
        )

    # max pool
    with tf.name_scope('pool2') as scope:
        pool2 = tf.layers.max_pooling2d(
            concat1,  # Input
            pool_size=(3, 3),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='VALID',  # "same" padding
            name='pool2'
        )

    # conv layer 3
    with tf.name_scope('conv3.1') as scope:
        conv3 = _conv2d_batch_norm(pool2, filters=192, padding="SAME", name="3.1")

    with tf.name_scope('conv3.2') as scope:
        conv3 = _conv2d_batch_norm(conv3, filters=192, padding="SAME", name="3.2")

    # max pool
    with tf.name_scope('pool3') as scope:
        pool3 = tf.layers.max_pooling2d(
            conv3,  # Input
            pool_size=(3, 3),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='VALID',  # "same" padding
            name='pool3'
        )

    # conv layer 4
    with tf.name_scope('conv4.1') as scope:
        conv4 = _conv2d_batch_norm(pool3, filters=256, padding="SAME", name="4.1")

    with tf.name_scope('conv4.2') as scope:
        conv4 = _conv2d_batch_norm(conv4, filters=256, padding="SAME", name="4.2")

    # max pool
    with tf.name_scope('pool4') as scope:
        pool4 = tf.layers.max_pooling2d(
            conv4,  # Input
            pool_size=(3, 3),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='VALID',  # "same" padding
            name='pool4'
        )

    # conv layer 5
    with tf.name_scope('conv5.1') as scope:
        conv5 = _conv2d_batch_norm(pool4, filters=256, padding="SAME", name="5.1")

    with tf.name_scope('conv5.2') as scope:
        conv5 = _conv2d_batch_norm(conv5, filters=256, padding="SAME", name="5.2")

    # max pool
    with tf.name_scope('pool5') as scope:
        pool5 = tf.layers.max_pooling2d(
            conv5,  # Input
            pool_size=(3, 3),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='VALID',  # "same" padding
            name='pool5'
        )

    # Flatten output
    with tf.name_scope('flatten') as scope:
        flat_output = tf.contrib.layers.flatten(pool5)

        # dropout at fc rate
        flat_output = tf.layers.dropout(flat_output, rate=fcdropout_rate, seed=116, training=training)

    # Fully connected layer 1
    with tf.name_scope('fc1') as scope:
        fc1 = tf.layers.dense(
            flat_output,
            2048,
            activation=None,
            kernel_initializer=tf.variance_scaling_initializer(scale=2, seed=117),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamF),
            name="fc1"
        )

        bn_fc1 = tf.layers.batch_normalization(
            fc1,
            axis=-1,
            momentum=0.9,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_fc1'
        )

        fc1_relu = tf.nn.relu(bn_fc1, name='fc1_relu')

        # dropout
        fc1_relu = tf.layers.dropout(fc1_relu, rate=fcdropout_rate, seed=118, training=training)

    # Fully connected layer 2
    with tf.name_scope('fc2') as scope:
        fc2 = tf.layers.dense(
            fc1_relu,  # input
            2048,  # 1024 hidden units
            activation=None,  # None
            kernel_initializer=tf.variance_scaling_initializer(scale=2, seed=119),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamF),
            name="fc2"
        )

        bn_fc2 = tf.layers.batch_normalization(
            fc2,
            axis=-1,
            momentum=0.9,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_fc2'
        )

        fc2_relu = tf.nn.relu(bn_fc2, name='fc2_relu')

        # dropout
        fc2_relu = tf.layers.dropout(fc2_relu, rate=fcdropout_rate, seed=120, training=training)

    # Output layer
    logits = tf.layers.dense(
        fc2_relu,
        num_classes,      # One output unit per category
        activation=None,  # No activation function
        kernel_initializer=tf.variance_scaling_initializer(scale=1, seed=121),
        bias_initializer=tf.zeros_initializer(),
        name="logits"
    )

    with tf.variable_scope('conv_1.0', reuse=True):
        conv_kernels1 = tf.get_variable('kernel')
        kernel_transposed = tf.transpose(conv_kernels1, [3, 0, 1, 2])

    with tf.variable_scope('visualization'):
        tf.summary.image('conv1.0/filters', kernel_transposed, max_outputs=32, collections=["kernels"])

    ## Loss function options
    # Regular mean cross entropy
    #mean_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

    # Different weighting method
    # This will weight the positive examples higher so as to improve recall
    weights = tf.multiply(3, tf.cast(tf.equal(y, 1), tf.int32)) + 1
    mean_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits, weights=weights))

    # Add in l2 loss
    loss = mean_ce + tf.losses.get_regularization_loss()

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Minimize cross-entropy
    train_op = optimizer.minimize(loss, global_step=global_step)

    # get the probabilites from the logits
    probabilities = tf.nn.softmax(logits, name="probabilities")

    # Compute predictions and accuracy from the probabilities
    predictions = tf.argmax(probabilities, axis=1, output_type=tf.int64)
    is_correct = tf.equal(y, predictions)

    accuracy, acc_op = tf.metrics.accuracy(
        labels=y,
        predictions=predictions,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        #metrics_collections="summaries",
        name="accuracy",
    )

    # calculate recall
    if num_classes > 2:
        recall = [0] * num_classes
        rec_op = [[]] * num_classes

        precision = [0] * num_classes
        prec_op = [[]] * num_classes

        for k in range(num_classes):
            recall[k], rec_op[k] = tf.metrics.recall(
                labels=tf.equal(y, k),
                predictions=tf.equal(predictions, k),
                updates_collections=tf.GraphKeys.UPDATE_OPS,
                metrics_collections=["summaries"]
            )

            precision[k], prec_op[k] = tf.metrics.precision(
                labels=tf.equal(y, k),
	                predictions=tf.equal(predictions, k),
                updates_collections=tf.GraphKeys.UPDATE_OPS,
                metrics_collections=["summaries"]
            )

            f1_score = 2 * ((precision * recall) / (precision + recall))
    else:
        recall, rec_op = tf.metrics.recall(labels=y, predictions=predictions, updates_collections=tf.GraphKeys.UPDATE_OPS, name="recall")
        precision, prec_op = tf.metrics.precision(labels=y, predictions=predictions, updates_collections=tf.GraphKeys.UPDATE_OPS, name="precision")
        f1_score = 2 * ( (precision * recall) / (precision + recall))

        #auc, auc_op = tf.metrics.auc(labels=y, predictions=probabilities[:,1], num_thresholds=50, name="auc_1", updates_collections=tf.GraphKeys.UPDATE_OPS)

        #tf.summary.scalar('auc_', auc, collections=["summaries"])

    # Create summary hooks
    tf.summary.scalar('accuracy', accuracy, collections=["summaries"])
    tf.summary.scalar('recall_1', recall, collections=["summaries"])
    tf.summary.scalar('cross_entropy', mean_ce, collections=["summaries"])
    #tf.summary.scalar('loss', loss, collections=["summaries"])
    tf.summary.scalar('learning_rate', learning_rate, collections=["summaries"])

    _, update_op = summary_lib.pr_curve_streaming_op(name='pr_curve',
                                                     predictions=probabilities[:,1],
                                                     labels=y,
                                                     updates_collections=tf.GraphKeys.UPDATE_OPS,
													 #metrics_collections=["summaries"],
                                                     num_thresholds=20)
    if num_classes == 2:
        tf.summary.scalar('precision_1', precision, collections=["summaries"])
        tf.summary.scalar('f1_score', f1_score, collections=["summaries"])

    # add this so that the batch norm gets run
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Merge all the summaries
    merged = tf.summary.merge_all()
    kernel_summaries = tf.summary.merge_all("kernels")

    print("Graph created...")
# ## Train

## CONFIGURE OPTIONS
if os.path.exists(os.path.join("model", model_name + '.ckpt.index')):
    init = False
else:
    init = True

meta_data_every = 1
log_to_tensorboard = True
print_every = 5  # how often to print metrics
checkpoint_every = 3  # how often to save model in epochs
use_gpu = False  # whether or not to use the GPU
print_metrics = True  # whether to print or plot metrics, if False a plot will be created and updated every epoch

# Placeholders for metrics
valid_acc_values = []
train_acc_values = []

config = tf.ConfigProto()

## train the model
with tf.Session(graph=graph, config=config) as sess:
    if log_to_tensorboard:
        train_writer = tf.summary.FileWriter('./logs/tr_' + model_name, sess.graph)
        test_writer = tf.summary.FileWriter('./logs/te_' + model_name)
    
    if not print_metrics:
        # create a plot to be updated as model is trained
        f, ax = plt.subplots(1,4,figsize=(24,5))
    
    # create the saver
    saver = tf.train.Saver()
    
    # If the model is new initialize variables, else restore the session
    if init:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, './model/'+model_name+'.ckpt')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print("Training model", model_name,"...")
    
    for epoch in range(epochs):
        sess.run(tf.local_variables_initializer())

        for i in range(steps_per_epoch):
            # Accuracy values (train) after each batch
            batch_acc = []
            batch_cost = []

            # create the metadata
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # Run training op and update ops
            if (i % 50 != 0) or (i == 0):
                # log the kernel images once per epoch
                if (i == (steps_per_epoch - 1)) and log_to_tensorboard:
                    _, _, _, image_summary, step = sess.run(
                        [train_op, extra_update_ops, update_op, kernel_summaries, global_step],
                        feed_dict={
                            training: True,
                        },
                        options=run_options,
                        run_metadata=run_metadata)

                    # write the summary
                    train_writer.add_summary(image_summary, step)
                else:
                    _, _, _, step = sess.run(
                        [train_op, extra_update_ops, update_op, global_step],
                            feed_dict={
                                training: True,
                            },
                            options=run_options,
                            run_metadata=run_metadata)

            # every 50th step get the metrics
            else:
                _, _, _, precision_value, summary, acc_value, cost_value, recall_value, step = sess.run(
                    [train_op, extra_update_ops, update_op, prec_op, merged, accuracy, mean_ce, rec_op, global_step],
                    feed_dict={
                        training: True,
                    },
                    options=run_options,
                    run_metadata=run_metadata)

                # Save accuracy (current batch)
                batch_acc.append(acc_value)
                batch_cost.append(cost_value)

                # log the summaries to tensorboard every 50 steps
                if log_to_tensorboard:
                    # write the summary
                    train_writer.add_summary(summary, step)

            # only log the meta data once per epoch
            if i == 1:
                train_writer.add_run_metadata(run_metadata, 'step %d' % step)
                
        # save checkpoint every nth epoch
        if(epoch % checkpoint_every == 0):
            print("Saving checkpoint")
            save_path = saver.save(sess, './model/'+model_name+'.ckpt')
    
            # Now that model is saved set init to false so we reload it next time
            init = False
        
        # init batch arrays
        batch_cv_acc = []

        # initialize the local variables so we have metrics only on the evaluation
        sess.run(tf.local_variables_initializer())

        print("Evaluating model...")
        # load the test data
        X_cv, y_cv = load_validation_data(percentage=1, how="normal")

        # evaluate the test data
        for X_batch, y_batch in get_batches(X_cv, y_cv, batch_size, distort=False):
            _, _, valid_acc, valid_recall, valid_precision, valid_fscore, valid_cost = sess.run(
                [update_op, extra_update_ops, accuracy, rec_op, prec_op, f1_score, mean_ce],
                feed_dict={
                    X: X_batch,
                    y: y_batch,
                    training: False
                })

            batch_cv_acc.append(valid_acc)

        # Write average of validation data to summary logs
        if log_to_tensorboard:
            # evaluate once more to get the summary, which will then be written to tensorboard
            summary, cv_accuracy = sess.run(
                [merged, accuracy],
                feed_dict={
                    X: X_cv[0:2],
                    y: y_cv[0:2],
                    training: False
                })

            test_writer.add_summary(summary, step)
        step += 1

        # delete the test data to save memory
        del (X_cv)
        del (y_cv)

        print("Done evaluating...")

        # take the mean of the values to add to the metrics
        valid_acc_values.append(np.mean(batch_cv_acc))
        train_acc_values.append(np.mean(batch_acc))

        # Print progress every nth epoch to keep output to reasonable amount
        if (epoch % print_every == 0):
            print(
            'Epoch {:02d} - step {} - cv acc: {:.3f} - train acc: {:.3f} (mean)'.format(
                epoch, step, np.mean(batch_cv_acc), np.mean(batch_acc)
            ))

        # Print data every 50th epoch so I can write it down to compare models
        if (not print_metrics) and (epoch % 50 == 0) and (epoch > 1):
            if (epoch % print_every == 0):
                print(
                'Epoch {:02d} - step {} - cv acc: {:.4f} - train acc: {:.3f} (mean)'.format(
                    epoch, step, np.mean(batch_cv_acc), np.mean(batch_acc)
                ))

    # stop the coordinator
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)

    ## Evaluate on test data
    X_te, y_te = load_validation_data(how="normal", data="test")

    test_accuracy = []
    test_recall = []
    test_predictions = []
    ground_truth = []

    # evaluate the test data
    for X_batch, y_batch in get_batches(X_te, y_te, batch_size, distort=False):
        yhat, test_acc_value, test_recall_value = sess.run([predictions, accuracy, rec_op], feed_dict=
        {
            X: X_batch,
            y: y_batch,
            training: False
        })

        test_accuracy.append(test_acc_value)
        test_recall.append(test_recall_value)
        test_predictions.append(yhat)
        ground_truth.append(y_batch)

    # print the results
    print("Mean Test Accuracy:", np.mean(test_accuracy))
    print("Mean Test Recall:", np.mean(test_recall))

    # save the predictions and truth for review
    np.save(os.path.join("data", "predictions_" + model_name + ".npy"), test_predictions)
    np.save(os.path.join("data", "truth_" + model_name + ".npy"), ground_truth)
