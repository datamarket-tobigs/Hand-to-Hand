import os
import glob
import numpy as np
import tensorflow as tf

from tqdm import tqdm
# from tf_utils import allow_memory_growth

# TFRecords
# http://solarisailab.com/archives/2603
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_list_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# serialize 직렬화? byte 형태로 변환 -> TFRecords
def serialize_example(image_str, label):
    feature = {
        'label': _float_list_feature(label),
        'image': _bytes_feature(image_str),
    }

    # Create a Features message using tf.train.Example.
    # tf.train.Example을 이용해서 Feature messeage를 생성합니다.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString() # SerializeToString 함수를 이용해서 binary string으로 변환


def tf_serialize_example(image_str, label):
    tf_string = tf.py_function(
        serialize_example,
        (image_str, label),
        tf.string)
    return tf.reshape(tf_string, ())  # The result is a scalar


def parse_fn(image_fn, labels, res):
    # load image: [res, res, 3]
    image = tf.io.read_file(image_fn)
    # for decode_bmp, decode_gif, decode_jpeg, and decode_png
    # to convert the input bytes string into a Tensor of type dtype.
    image = tf.io.decode_image(image, channels=3, dtype=tf.uint8)
    image.set_shape([None, None, 3]) # 3차원 RGB
    image = tf.image.resize(image, size=[res, res]) # resize!
    image.set_shape([res, res, 3])
    image = tf.cast(image, dtype=tf.uint8)
    image_str = tf.image.encode_png(image)

    # set labels
    return image_str, labels


def create_tfrecord_data(input_data, output_fn, res):
    image_fns = input_data['image_fns']
    labels = input_data['labels']

    dataset = tf.data.Dataset.from_tensor_slices((image_fns, labels))
    dataset = dataset.map(lambda f, l: parse_fn(f, l, res), num_parallel_calls=8) # image_fn -> image_str
    dataset = dataset.map(lambda s, l: tf_serialize_example(s, l), num_parallel_calls=8)
    # TFRecords 쓰기
    writer = tf.data.experimental.TFRecordWriter(output_fn)
    writer.write(dataset)
    return

def raw_data_to_npy(txt_path, parse_first_line=False):
    with open(txt_path) as f:
        raw_data = f.readlines()

    if parse_first_line:
        raw_data = [rd.split(',') for rd in raw_data]
    else:
        raw_data = [rd.split(',')[1:] for rd in raw_data]
    
    npy_data = np.array(raw_data).astype(np.float)
    return npy_data



def main():
    # allow_memory_growth() # 메모리 증가를 허용

    # prepare variables
    res = 512
    divide = 500
    data_base_dir = '../data'
    dst_tfrecord_dir = os.path.join(data_base_dir, 'tfrecords')
    if not os.path.exists(dst_tfrecord_dir):
        os.makedirs(dst_tfrecord_dir)

    # load image file names
    image_fns = glob.glob(os.path.join(data_base_dir, 'resize_image', '*.png')) # .png 파일 다
    image_fns = sorted(image_fns)

    # load labels
    label_fn = os.path.join(data_base_dir, '키포인트txt', 'label.txt')
    label_fn = raw_data_to_npy(label_fn)

    # start converting
    n_total = len(image_fns)
    interval_list = [(v * divide, (v + 1) * divide) for v in range(n_total // divide)]
    for fn_idx, (start_idx, end_idx) in enumerate(tqdm(interval_list)):
        output_fn = os.path.join(dst_tfrecord_dir, f'{fn_idx:04d}.tfrecord') # 500개씩 0001.tfrecord, 0002.tfrecord 식

        sliced_data = {
            'image_fns': image_fns[start_idx:end_idx],
            'labels': label_fn[start_idx:end_idx],
        }
        create_tfrecord_data(sliced_data, output_fn, res)
    print('DONE!!')
    return


if __name__ == '__main__':
    main()