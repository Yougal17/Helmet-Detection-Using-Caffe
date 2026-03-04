import os
import cv2

TEST_LIST = '/workspace/data/test.txt'
IMAGES_DIR = '/workspace/data/images'
OUTPUT_FILE = '/workspace/data/test_name_size.txt'

with open(TEST_LIST) as f:
    names = [l.strip() for l in f if l.strip()]

with open(OUTPUT_FILE, 'w') as out:
    for name in names:
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(IMAGES_DIR, name + ext)
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    h, w = img.shape[:2]
                    out.write('{} {} {}\n'.format(name, h, w))
                break

print("Done! Written {} entries to test_name_size.txt".format(len(names)))