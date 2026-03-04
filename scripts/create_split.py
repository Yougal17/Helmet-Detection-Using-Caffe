import os
import random

images_dir = '/workspace/data/images'
output_dir = '/workspace/data'

# Get all image names without extension
all_images = [os.path.splitext(f)[0] for f in os.listdir(images_dir) 
              if f.endswith('.png') or f.endswith('.jpg')]

random.seed(42)
random.shuffle(all_images)

split = int(0.8 * len(all_images))
train = all_images[:split]
test  = all_images[split:]

with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
    for name in train:
        f.write(name + '\n')

with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
    for name in test:
        f.write(name + '\n')

print("Total images : {}".format(len(all_images)))
print("Train images : {}".format(len(train)))
print("Test  images : {}".format(len(test)))
