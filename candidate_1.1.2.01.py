import numpy as np
import os
import wget
from sklearn.model_selection import train_test_split
import tensorflow as tf
from training_utils import download_file, get_batches, read_and_decode_single_example, load_validation_data, \
    download_data, evaluate_model, get_training_data, _conv2d_batch_norm, _dense_batch_norm
import argparse
from tensorboard import summary as summary_lib

# download the data
download_data()
# ## Create Model

# config
# If number of epochs has been passed in use that, otherwise default to 50
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epochs to train", type=int)
args = parser.parse_args()

if args.epochs:
    epochs = args.epochs
else:
    epochs = 50

batch_size = 64

train_files, total_records = get_training_data(type="new")

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
lamF = 0.00250

# use dropout
dropout = True
fcdropout_rate = 0.7
convdropout_rate = 0.01
pooldropout_rate = 0.25

num_classes = 2

## Build the graph
graph = tf.Graph()

# whether to retrain model from scratch or use saved model
init = True
model_name = "model_s1.1.2.03"
# 1.1.2.01 - trying to mimic some features of VGG
# 1.1.2.02 - changed conv1 to stride 2, otherwise used too much memory

with graph.as_default():
    training = tf.placeholder(dtype=tf.bool, name="is_training")
    is_testing = tf.placeholder(dtype=bool, shape=(), name="is_testing")

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

    # Input stem
    conv1 = _conv2d_batch_norm(X, filters=32, stride=(1, 1), training=training, padding="VALID", name="1.1")

    # Max pooling layer 1
    with tf.name_scope('pool1') as scope:
        pool1 = tf.layers.max_pooling2d(
            conv1,  # Input
            pool_size=(2, 2),
            strides=(2, 2),
            padding='VALID',
            name='pool1'
        )

        # optional dropout
        if dropout:
            pool1 = tf.layers.dropout(pool1, rate=pooldropout_rate, seed=103, training=training)

    # Layer 2 branch 1
    conv2 = _conv2d_batch_norm(pool1, filters=64, stride=(1, 1), training=training, padding="SAME", name="2.1")
    conv2 = _conv2d_batch_norm(conv2, filters=64, stride=(1, 1), training=training, padding="SAME", name="2.2")

    # Layer 2 branch 2
    conv21 = _conv2d_batch_norm(pool1, filters=64, stride=(1, 1), training=training, padding="SAME", name="2.3")

    # Concat 2
    with tf.name_scope("concat2") as scope:
        concat2 = tf.concat(
            [conv2, conv21],
            axis=3,
            name='concat2'
        )

    # Pool 2
    with tf.name_scope('pool2') as scope:
        pool2 = tf.layers.max_pooling2d(
            concat2,
            pool_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
            name='pool2'
        )

    # Layer 3
    conv3 = _conv2d_batch_norm(pool2, filters=96, kernel_size=(1,1), stride=(1, 1), training=training, padding="SAME", name="3.0")
    conv3 = _conv2d_batch_norm(conv3, filters=128, stride=(1, 1), training=training, padding="SAME", name="3.1")
    conv3 = _conv2d_batch_norm(conv3, filters=128, stride=(1, 1), training=training, padding="SAME", name="3.2")
    conv3 = _conv2d_batch_norm(conv3, filters=128, stride=(1, 1), training=training, padding="SAME", name="3.3")

    # Max pooling layer 3
    with tf.name_scope('pool3') as scope:
        pool3 = tf.layers.max_pooling2d(
            conv3,  # Input
            pool_size=(2, 2),  # Pool size: 2x2
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='pool3'
        )

        if dropout:
            pool3 = tf.layers.dropout(pool3, rate=pooldropout_rate, seed=109, training=training)

    # Layer 4
    conv4 = _conv2d_batch_norm(pool3, filters=256, stride=(1, 1), training=training, padding="SAME", name="4.1")
    conv4 = _conv2d_batch_norm(conv4, filters=256, stride=(1, 1), training=training, padding="SAME", name="4.2")

    # Max pooling layer 4
    with tf.name_scope('pool4') as scope:
        pool4 = tf.layers.max_pooling2d(
            conv4,  # Input
            pool_size=(2, 2),  # Pool size: 2x2
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='pool4'
        )

        if dropout:
            pool4 = tf.layers.dropout(pool4, rate=pooldropout_rate, seed=112, training=training)

    # Layer 5
    conv5 = _conv2d_batch_norm(pool4, filters=384, stride=(1, 1), training=training, padding="SAME", name="5.1")
    conv5 = _conv2d_batch_norm(conv5, filters=384, stride=(1, 1), training=training, padding="SAME", name="5.2")

    # Max pooling layer 4
    with tf.name_scope('pool5') as scope:
        pool5 = tf.layers.max_pooling2d(
            conv5,
            pool_size=(2, 2),  # Pool size: 2x2
            strides=(2, 2),  # Stride: 2
            padding='SAME',
            name='pool5'
        )

        if dropout:
            pool5 = tf.layers.dropout(pool5, rate=pooldropout_rate, seed=115, training=training)

    # Flatten output
    with tf.name_scope('flatten') as scope:
        flat_output = tf.contrib.layers.flatten(pool5)

        # dropout at fc rate
        flat_output = tf.layers.dropout(flat_output, rate=fcdropout_rate, seed=116, training=training)

    # Fully connected layer 1
    fc1 = _dense_batch_norm(flat_output, 1024, training=training, epsilon=1e-8, dropout_rate=fcdropout_rate, lambd=lamF, name="1.1")
    fc2 = _dense_batch_norm(fc1, 512, training=training, epsilon=1e-8, dropout_rate=fcdropout_rate, lambd=lamF, name="1.2")
    fc3 = _dense_batch_norm(fc2, 256, training=training, epsilon=1e-8, dropout_rate=fcdropout_rate, lambd=lamF, name="1.3")

    # Output layer
    logits = tf.layers.dense(
        fc3,
        num_classes,  # One output unit per category
        activation=None,  # No activation function
        kernel_initializer=tf.variance_scaling_initializer(scale=1, seed=121),
        bias_initializer=tf.zeros_initializer(),
        name="logits"
    )

    with tf.variable_scope('conv_1.1', reuse=True):
        conv_kernels1 = tf.get_variable('kernel')
        kernel_transposed = tf.transpose(conv_kernels1, [3, 0, 1, 2])

    with tf.variable_scope('visualization'):
        tf.summary.image('conv_1.1/filters', kernel_transposed, max_outputs=32, collections=["kernels"])

    ## Loss function options
    # Regular mean cross entropy
    #mean_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

    # This will weight the positive examples higher so as to improve recall
    weights = tf.multiply(2, tf.cast(tf.equal(y, 1), tf.int32)) + 1
    mean_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits, weights=weights))

    # Add in l2 loss
    loss = mean_ce + tf.losses.get_regularization_loss()

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Minimize cross-entropy
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Compute predictions and accuracy
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    is_correct = tf.equal(y, predictions)

    accuracy, acc_op = tf.metrics.accuracy(
        labels=y,
        predictions=predictions,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        #metrics_collections="summaries",
        name="accuracy",
    )

    # get the probabilites for the classes
    probabilities = tf.nn.softmax(logits, name="probabilities")

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
        f, ax = plt.subplots(1, 4, figsize=(24, 5))

    # create the saver
    saver = tf.train.Saver()

    # If the model is new initialize variables, else restore the session
    if init:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, './model/' + model_name + '.ckpt')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print("Training model", model_name, "...")

    for epoch in range(epochs):
        sess.run(tf.local_variables_initializer())

        # Accuracy values (train) after each batch
        batch_acc = []
        batch_cost = []

        for i in range(steps_per_epoch):
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
        if (epoch % checkpoint_every == 0):
            print("Saving checkpoint")
            save_path = saver.save(sess, './model/' + model_name + '.ckpt')

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


