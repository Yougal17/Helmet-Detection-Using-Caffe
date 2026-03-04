import sys
sys.path.insert(0, '/ssd-caffe/python')
import caffe
import cv2
import numpy as np
import os

# Paths
MODEL_DEF    = '/workspace/models/deploy.prototxt'
MODEL_WEIGHTS = '/workspace/output/snapshots/helmet_ssd_iter_1800.caffemodel'
TEST_LIST    = '/workspace/data/test.txt'
IMAGES_DIR   = '/workspace/data/images'
OUTPUT_DIR   = '/workspace/output/detections'

CLASSES = ['background', 'With Helmet', 'Without Helmet']
COLORS  = [(0,0,0), (0,255,0), (0,0,255)]  # green=helmet, red=no helmet

IMAGE_SIZE = 300
CONF_THRESHOLD = 0.2
NMS_THRESHOLD = 0.01

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

caffe.set_mode_cpu()
net = caffe.Net(MODEL_DEF, MODEL_WEIGHTS, caffe.TEST)

def preprocess(img):
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32)
    img -= np.array([104, 117, 123], dtype=np.float32)
    img = img * 0.007843
    img = img.transpose(2, 0, 1)
    return img

def detect(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    orig = img.copy()
    h, w = img.shape[:2]
    blob = preprocess(img)
    net.blobs['data'].reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    net.blobs['data'].data[...] = blob
    detections = net.forward()['detection_out']

    # Group by class and apply NMS
    class_boxes = {}
    for i in range(detections.shape[2]):
        det = detections[0, 0, i]
        conf = float(det[2])
        if conf < CONF_THRESHOLD:
            continue
        label = int(det[1])
        if label == 0:
            continue
        xmin = int(det[3] * w)
        ymin = int(det[4] * h)
        xmax = int(det[5] * w)
        ymax = int(det[6] * h)
        if label not in class_boxes:
            class_boxes[label] = []
        class_boxes[label].append([xmin, ymin, xmax, ymax, conf])

    results = []
    for label, boxes in class_boxes.items():
        boxes_arr = np.array([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=np.float32)
        scores = np.array([b[4] for b in boxes], dtype=np.float32)
        indices = cv2.dnn.NMSBoxes(
            boxes_arr.tolist(), scores.tolist(),
            CONF_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            for idx in indices:
                if isinstance(idx, (list, np.ndarray)):
                    idx = idx[0]
                b = boxes[idx]
                results.append((label, b[4], b[0], b[1], b[2], b[3]))
    return orig, results
def draw_detections(img, results):
    for (label, conf, xmin, ymin, xmax, ymax) in results:
        color = COLORS[label] if label < len(COLORS) else (255,255,0)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        text = '{}: {:.2f}'.format(CLASSES[label], conf)
        cv2.putText(img, text, (xmin, ymin-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

# Run on first 10 test images
with open(TEST_LIST) as f:
    test_names = [l.strip() for l in f if l.strip()][:10]

saved = 0
for name in test_names:
    img_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        p = os.path.join(IMAGES_DIR, name + ext)
        if os.path.exists(p):
            img_path = p
            break
    if img_path is None:
        continue

    orig, results = detect(img_path)
    if orig is None:
        continue

    print('Image: {} | Detections: {}'.format(name, len(results)))
    for (label, conf, xmin, ymin, xmax, ymax) in results:
        print('  {} {:.2f} [{},{},{},{}]'.format(
            CLASSES[label], conf, xmin, ymin, xmax, ymax))

    out_img = draw_detections(orig, results)
    out_path = os.path.join(OUTPUT_DIR, name + '_detected.jpg')
    cv2.imwrite(out_path, out_img)
    saved += 1

print('\nSaved {} detection images to {}'.format(saved, OUTPUT_DIR))