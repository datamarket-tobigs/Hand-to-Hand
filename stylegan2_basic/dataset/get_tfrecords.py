import os
import glob
import tensorflow as tf


def parse_tfrecord(raw_record, res):
    feature_description = {
        'label': tf.io.FixedLenFeature([259], tf.float32),
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }

    # parse feature
    parsed = tf.io.parse_single_example(raw_record, feature_description)

    # labels
    labels = tf.reshape(parsed['label'], shape=[259])

    # image
    image = tf.io.decode_png(parsed['image'])
    image = tf.image.resize(image, size=[res, res])
    image = tf.transpose(image, perm=[2, 0, 1])
    image = tf.cast(image, dtype=tf.dtypes.float32)
    image = image / 127.5 - 1.0
    image.set_shape([3, res, res])
    return image, labels


def input_fn(filenames, res, batch_size, epochs):
    files = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x),
                               # cycle_length: number of files to read concurrently
                               # If you want to read from all your files(*.tfrecord) to create a batch,
                               # set this to the number of files
                               cycle_length=len(filenames),
                               # block_length: each time we read from a file, reads block_length elements from this file
                               block_length=1)
    dataset = dataset.map(lambda x: parse_tfrecord(x, res), num_parallel_calls=8)
    dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=None)
    return dataset


def get_dataset(tfrecord_dir, res, batch_size, is_train):
    # get dataset
    filenames = glob.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
    filenames = sorted(filenames)
    train_filenames = filenames[:-1]
    val_filenames = filenames[-1:]
    filenames = train_filenames if is_train else val_filenames

    # create dataset
    dataset = input_fn(filenames, res, batch_size=batch_size, epochs=None)
    return dataset


def get_label_depth():
    # define label spec
    label_depths = {
        'label_dim': 259,
        # 'label_dim': 258,
        # 'person_dim': 4,
        # 'keypoint_dim': 254,
    }
    return label_depths


def main():
    import numpy as np
    from PIL import Image

    # res = 256
    res = 512
    data_base_dir = '../data'
    folder_name = 'tfrecords'
    tfrecord_dir = os.path.join(data_base_dir, folder_name)

    batch_size = 2
    is_train = True

    # label_depths = get_label_depth()
    dataset = get_dataset(tfrecord_dir, res, batch_size, is_train)

    for ii in range(3): # 이미지, 라벨 3개만 출력
        for real_images, labels in dataset.take(1):
            image_raw = real_images.numpy()
            image_raw = image_raw[0]
            image_raw = np.transpose(image_raw, axes=[1, 2, 0])
            image_raw = (image_raw + 1.0) * 127.5  # -1~1 -> 0~255
            image = Image.fromarray(np.uint8(image_raw))
            image.show()
            print(labels[0])
    return


if __name__ == '__main__':
    main()
