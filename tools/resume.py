import os
import json
import hashlib
import re

from collections import defaultdict
from re import Pattern, Match
from typing import Dict, Generator, Tuple

import tqdm
import lxml.etree
import tensorflow as tf

from absl.flags import FLAGS
from absl import app, flags, logging
from tensorflow.core.example.example_pb2 import Example

flags.DEFINE_string("data_dir", None, "path to raw resum dataset")
flags.DEFINE_enum("split", "train", ["train", "val"], "specify train or val split")
flags.DEFINE_string("output_file", None, "output_dataset")
flags.DEFINE_string("classes", None, "classes file")


def load_json_file(path: str) -> Dict:
    with open(path, "r") as f:
        file = f.read()

    file_dict = json.loads(file)
    return file_dict


def read_file_path_in_dir(dir_path: str) -> Generator[str, None, None]:
    directory_path: Path = Path(dir_path)

    sub_path: Path
    for sub_path in directory_path.iterdir():
        if sub_path.is_dir():
            continue

        yield sub_path.absolute().__str__()


def get_name_from_path(path: str) -> Tuple[str, str]:
    pattern: Pattern = re.compile(r"(?<!\\/)(?P<value>\w+\.\w+)")
    matcher: Match = pattern.search(path)

    if not matcher:
        return "", ""

    file_name = matcher.group()
    name, suffix = file_name.split(".")

    return name, suffix


def build_example(file_name: str, img_info: Dict) -> Example:
    """
    :param file_name without suffix
    :param img_info: image info from json file
    :return: tf.train.Example
    """
    data: defaultdict = defaultdict(list)

    encoded_image_data: bytes = img_info["imageData"].encode("utf8")
    all_labels = img_info["shapes"]
    img_width: int = img_info["imageWidth"]
    img_height: int = img_info["imageHeight"]
    filename: bytes = f"{file_name}.jpg".encode("utf8")
    image_format: bytes = "jpg".encode("utf8")

    for label in all_labels:
        data["classes_text"].append(label["label"].encode("utf8"))
        data["classes"].append(label["group_id"])

        data["xmins"].append(label["points"][0][0] / img_width)
        data["xmaxs"].append(label["points"][1][0] / img_width)
        data["ymins"].append(label["points"][0][1] / img_height)
        data["ymaxs"].append(label["points"][1][1] / img_height)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_width])),
        'image/filename': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[filename])
        ),
        'image/source_id': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[filename])
        ),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=data["xmins"])),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=data["xmaxs"])),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=data["ymins"])),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=data["ymaxs"])),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=data["classes_text"])),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=data["classes"])),
    }))
    return tf_example


def main(_argv):
    del _argv

    print(FLAGS.data_dir)


if __name__ == '__main__':
    # app.run(main)
    path = "/home/mrj/Sundry/resume_labels/00001.json"
    j = load_json_file(path)
    file_name, _ = get_name_from_path(path)

    print(build_example(file_name, j))