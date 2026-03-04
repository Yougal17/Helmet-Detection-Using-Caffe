import sys
sys.path.insert(0, '/ssd-caffe/python')
import os
import lmdb
import cv2
import xml.etree.ElementTree as ET
import shutil
from caffe.proto import caffe_pb2

IMAGES_DIR = '/workspace/data/images'
ANNOT_DIR = '/workspace/data/annotations'
TRAIN_LIST = '/workspace/data/train.txt'
TEST_LIST = '/workspace/data/test.txt'
TRAIN_LMDB = '/workspace/data/lmdb/train_lmdb'
TEST_LMDB = '/workspace/data/lmdb/test_lmdb'

NAME_TO_LABEL = {
    'background': 0,
    'With Helmet': 1,
    'Without Helmet': 2,
}

IMAGE_SIZE = 300

def read_image_list(list_file):
    with open(list_file) as f:
        return [l.strip() for l in f if l.strip()]

def parse_annotation(xml_path, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        label = NAME_TO_LABEL.get(name, -1)
        if label < 0:
            print("WARNING: unknown class '{}' skipping".format(name))
            continue
        bb = obj.find('bndbox')
        xmin = float(bb.find('xmin').text) / img_w
        ymin = float(bb.find('ymin').text) / img_h
        xmax = float(bb.find('xmax').text) / img_w
        ymax = float(bb.find('ymax').text) / img_h
        objects.append((label, xmin, ymin, xmax, ymax))
    return objects

def create_lmdb(image_names, lmdb_path, desc):
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)

    env = lmdb.open(lmdb_path, map_size=int(1e10))
    success = 0
    skipped = 0

    with env.begin(write=True) as txn:
        for idx, name in enumerate(image_names):
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                p = os.path.join(IMAGES_DIR, name + ext)
                if os.path.exists(p):
                    img_path = p
                    break

            if img_path is None:
                print("Missing image: {}".format(name))
                skipped += 1
                continue

            xml_path = os.path.join(ANNOT_DIR, name + '.xml')
            if not os.path.exists(xml_path):
                print("Missing annotation: {}".format(name))
                skipped += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            # Fix grayscale or RGBA images
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            h, w = img.shape[:2]
            objects = parse_annotation(xml_path, w, h)
            if not objects:
                skipped += 1
                continue

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.astype('uint8')
            img = img.transpose(2, 0, 1)

            datum = caffe_pb2.AnnotatedDatum()
            datum.type = caffe_pb2.AnnotatedDatum.BBOX
            datum.datum.channels = 3
            datum.datum.height = IMAGE_SIZE
            datum.datum.width = IMAGE_SIZE
            datum.datum.data = img.tobytes()
            datum.datum.encoded = False

            group = datum.annotation_group.add()
            group.group_label = objects[0][0]

            for (label, xmin, ymin, xmax, ymax) in objects:
                ann = group.annotation.add()
                bbox = ann.bbox
                bbox.xmin = xmin
                bbox.ymin = ymin
                bbox.xmax = xmax
                bbox.ymax = ymax
                bbox.label = label

            key = '{:08d}_{}'.format(idx, name).encode('utf-8')
            txn.put(key, datum.SerializeToString())
            success += 1

            if success % 100 == 0:
                print("{}: processed {}/{}".format(desc, success, len(image_names)))

    print("{} done - saved: {}, skipped: {}".format(desc, success, skipped))

train_names = read_image_list(TRAIN_LIST)
test_names = read_image_list(TEST_LIST)

print("Creating TRAIN lmdb ({} images)...".format(len(train_names)))
create_lmdb(train_names, TRAIN_LMDB, 'TRAIN')

print("Creating TEST lmdb ({} images)...".format(len(test_names)))
create_lmdb(test_names, TEST_LMDB, 'TEST')

print("All done!")