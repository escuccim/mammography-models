import numpy as np
import os
import wget
from sklearn.cross_validation  import train_test_split
import tensorflow as tf

## download a file to a location in the data folder
def download_file(url, name):
    print("\nDownloading " + name + "...")

    # check that the data directory exists
    try:
        os.stat("data")
    except:
        os.mkdir("data")

    fname = wget.download(url, os.path.join('data', name))

## Batch generator
def get_batches(X, y, batch_size, distort=True):
    # Shuffle X,y
    shuffled_idx = np.arange(len(y))
    np.random.shuffle(shuffled_idx)
    i, h, w, c = X.shape

    # Enumerate indexes by steps of batch_size
    for i in range(0, len(y), batch_size):
        batch_idx = shuffled_idx[i:i + batch_size]
        X_return = X[batch_idx]

        # do random flipping of images
        coin = np.random.binomial(1, 0.5, size=None)
        if coin and distort:
            X_return = X_return[..., ::-1, :]

        yield X_return, y[batch_idx]


## read data from tfrecords file
def read_and_decode_single_example(filenames, label_type='label_normal', normalize=False, distort=False, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'label_normal': tf.FixedLenFeature([], tf.int64),
            'label_mass': tf.FixedLenFeature([], tf.int64),
            'label_benign': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        })

    # extract the data
    label = features[label_type]
    image = tf.decode_raw(features['image'], tf.uint8)

    # reshape and scale the image
    image = tf.reshape(image, [299, 299, 1])

    if normalize:
        image = tf.image.per_image_standardization(image)

    # random flipping of image
    if distort:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    #tf.cast(image, dtype=tf.float32)

    # return the image and the label
    return image, label


## load the test data from files
def load_validation_data(data="validation", how="class", percentage=0.5):
    if data == "validation":
        # load the two data files
        X_cv_0 = np.load(os.path.join("data", "test2_data_0.npy"))
        labels_0 = np.load(os.path.join("data", "test2_labels_0.npy"))

        X_cv_1 = np.load(os.path.join("data", "test2_data_1.npy"))
        labels_1 = np.load(os.path.join("data", "test2_labels_1.npy"))

        # concatenate them
        X_cv = np.concatenate([X_cv_0, X_cv_1], axis=0)
        labels = np.concatenate([labels_0, labels_1], axis=0)

        # delete the old files to save memory
        del (X_cv_0)
        del (X_cv_1)
        del (labels_0)
        del (labels_1)

    elif data == "test":
        X_cv = np.load(os.path.join("data", "mias_test_images.npy"))
        labels = np.load(os.path.join("data", "mias_test_labels_enc.npy"))

    # encode the labels appropriately
    if how == "class":
        y_cv = labels
    elif how == "normal":
        y_cv = np.zeros(len(labels))
        y_cv[labels != 0] = 1
    elif how == "mass":
        y_cv = np.zeros(len(labels))
        y_cv[labels == 1] = 1
        y_cv[labels == 3] = 1
        y_cv[labels == 2] = 2
        y_cv[labels == 4] = 4
    elif how == "benign":
        y_cv = np.zeros(len(labels))
        y_cv[labels == 1] = 1
        y_cv[labels == 2] = 1
        y_cv[labels == 3] = 2
        y_cv[labels == 4] = 2

    # shuffle the data
    X_cv, _, y_cv, _ = train_test_split(X_cv, y_cv, test_size=1 - percentage)

    return X_cv, y_cv


## evaluate the model to see the predictions
def evaluate_model(graph, config, model_name, how="normal", batch_size=32):
    X_te, y_te = load_validation_data(how=how, data="test")

    with tf.Session(graph=graph, config=config) as sess:
        # create the saver
        saver = tf.train.Saver()
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, './model/' + model_name + '.ckpt')

        test_accuracy = []
        test_recall = []
        test_predictions = []
        ground_truth = []

        # evaluate the test data
        for X_batch, y_batch in get_batches(X_te, y_te, batch_size // 2, distort=False):
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
    print("Mean Recall:", np.mean(test_recall))
    print("Mean Accuracy:", np.mean(test_accuracy))

    return test_accuracy, test_recall, test_predictions, ground_truth


## Download the data if it doesn't already exist
def download_data():
    # download main training tfrecords files
    if not os.path.exists(os.path.join("data", "training_0.tfrecords")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_0.tfrecords', 'training_0.tfrecords')

    if not os.path.exists(os.path.join("data", "training_1.tfrecords")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_1.tfrecords', 'training_1.tfrecords')

    if not os.path.exists(os.path.join("data", "training_2.tfrecords")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_2.tfrecords', 'training_2.tfrecords')

    if not os.path.exists(os.path.join("data", "training_3.tfrecords")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_3.tfrecords', 'training_3.tfrecords')

    # download validation data
    if not os.path.exists(os.path.join("data", "test2_labels_0.npy")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test2_labels_0.npy', 'test2_labels_0.npy')

    if not os.path.exists(os.path.join("data", "test2_labels_1.npy")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test2_labels_1.npy', 'test2_labels_1.npy')

    if not os.path.exists(os.path.join("data", "test2_data_0.npy")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test2_data_0.npy', 'test2_data_0.npy')

    if not os.path.exists(os.path.join("data", "test2_data_1.npy")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test2_data_1.npy', 'test2_data_1.npy')

    # download MIAS test data
    if not os.path.exists(os.path.join("data", "mias_test_images.npy")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/mias_test_images.npy', 'mias_test_images.npy')

    if not os.path.exists(os.path.join("data", "mias_test_labels_enc.npy")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/mias_test_labels_enc.npy', 'mias_test_labels_enc.npy')

    # download secondary training tfrecords files
    if not os.path.exists(os.path.join("data", "training2_0.tfrecords")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training2_0.tfrecords', 'training2_0.tfrecords')

    if not os.path.exists(os.path.join("data", "training2_1.tfrecords")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training2_1.tfrecords', 'training2_1.tfrecords')

    if not os.path.exists(os.path.join("data", "training2_2.tfrecords")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training2_2.tfrecords', 'training2_2.tfrecords')

    if not os.path.exists(os.path.join("data", "training_3.tfrecords")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_3.tfrecords', 'training_3.tfrecords')


def train(config, graph, log_to_tensorboard=True, print_metrics=True, epochs=50, checkpoint_every=1):

    # initialize containers for the metrics
    valid_acc_values = []
    valid_recall_values = []
    valid_cost_values = []
    train_acc_values = []
    train_recall_values = []
    train_cost_values = []
    train_lr_values = []
    train_loss_values = []

    # start the session
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

        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("Training model", model_name, "...")

        for epoch in range(epochs):
            for i in range(steps_per_epoch):
                # Accuracy values (train) after each batch
                batch_acc = []
                batch_cost = []
                batch_loss = []
                batch_lr = []
                batch_recall = []

                # create the metadata
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                # Run training and evaluate accuracy
                _, _, precision_value, summary, acc_value, cost_value, loss_value, recall_value, step, lr = sess.run(
                    [train_op, extra_update_ops, prec_op,
                     merged, accuracy, mean_ce, loss, rec_op, global_step,
                     learning_rate], feed_dict={
                        # X: X_batch,
                        # y: y_batch,
                        training: True,
                        is_testing: False,
                    },
                    options=run_options,
                    run_metadata=run_metadata)

                # Save accuracy (current batch)
                batch_acc.append(acc_value)
                batch_cost.append(cost_value)
                batch_lr.append(lr)
                batch_loss.append(loss_value)
                batch_recall.append(np.mean(recall_value))

                # write the summary
                if log_to_tensorboard:
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
            batch_cv_cost = []
            batch_cv_loss = []
            batch_cv_recall = []
            batch_cv_precision = []
            batch_cv_fscore = []

            ## evaluate on test data if it exists, otherwise ignore this step
            if evaluate:
                print("Evaluating model...")
                # load the test data
                X_cv, y_cv = load_validation_data(percentage=1, how="normal")

                # evaluate the test data
                for X_batch, y_batch in get_batches(X_cv, y_cv, batch_size // 2, distort=False):
                    summary, valid_acc, valid_recall, valid_precision, valid_fscore, valid_cost, valid_loss = sess.run(
                        [merged, accuracy, rec_op, prec_op, f1_score, mean_ce, loss],
                        feed_dict={
                            X: X_batch,
                            y: y_batch,
                            is_testing: True,
                            training: False
                        })

                    batch_cv_acc.append(valid_acc)
                    batch_cv_cost.append(valid_cost)
                    batch_cv_loss.append(valid_loss)
                    batch_cv_recall.append(np.mean(valid_recall))
                    batch_cv_precision.append(np.mean(valid_precision))

                    # the first fscore will be nan so don't add that one
                    if not np.isnan(valid_fscore):
                        batch_cv_fscore.append(np.mean(valid_fscore))

                # Write average of validation data to summary logs
                if log_to_tensorboard:
                    summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=np.mean(batch_cv_acc)),
                                                tf.Summary.Value(tag="cross_entropy",
                                                                 simple_value=np.mean(batch_cv_cost)),
                                                tf.Summary.Value(tag="recall_1", simple_value=np.mean(batch_cv_recall)),
                                                tf.Summary.Value(tag="precision_1",
                                                                 simple_value=np.mean(batch_cv_precision)),
                                                tf.Summary.Value(tag="f1_score", simple_value=np.mean(batch_cv_fscore)),
                                                ])

                    test_writer.add_summary(summary, step)
                    step += 1

                # delete the test data to save memory
                del (X_cv)
                del (y_cv)

                print("Done evaluating...")
            else:
                batch_cv_acc.append(0)
                batch_cv_cost.append(0)
                batch_cv_loss.append(0)
                batch_cv_recall.append(0)

            # take the mean of the values to add to the metrics
            valid_acc_values.append(np.mean(batch_cv_acc))
            valid_cost_values.append(np.mean(batch_cv_cost))
            train_acc_values.append(np.mean(batch_acc))
            train_cost_values.append(np.mean(batch_cost))
            train_lr_values.append(np.mean(batch_lr))
            train_loss_values.append(np.mean(batch_loss))
            train_recall_values.append(np.mean(batch_recall))
            valid_recall_values.append(np.mean(batch_cv_recall))

            # Print progress every nth epoch to keep output to reasonable amount
            if (epoch % print_every == 0):
                print(
                    'Epoch {:02d} - step {} - cv acc: {:.3f} - train acc: {:.3f} (mean) - cv cost: {:.3f} - lr: {:.5f}'.format(
                        epoch, step, np.mean(batch_cv_acc), np.mean(batch_acc), np.mean(batch_cv_cost), lr
                    ))

            # Print data every 50th epoch so I can write it down to compare models
            if (not print_metrics) and (epoch % 50 == 0) and (epoch > 1):
                if (epoch % print_every == 0):
                    print(
                        'Epoch {:02d} - step {} - cv acc: {:.4f} - train acc: {:.3f} (mean) - cv cost: {:.3f} - lr: {:.5f}'.format(
                            epoch, step, np.mean(batch_cv_acc), np.mean(batch_acc), np.mean(batch_cv_cost), lr
                        ))

                    # stop the coordinator
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)

        test_accuracy, test_recall, test_predictions, ground_truth = evaluate_model(graph, config, model_name)

        # save the predictions and truth for review
        np.save(os.path.join("data", "predictions_" + model_name + ".npy"), test_predictions)
        np.save(os.path.join("data", "truth_" + model_name + ".npy"), ground_truth)