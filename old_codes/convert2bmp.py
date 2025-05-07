import os
from PIL import Image

source_dir = '/home/jari/PycharmProjects/koodaus/stegaexpose/ml/clean.jpg'
target_dir = '/home/jari/PycharmProjects/koodaus/stegaexpose/ml/clean'

os.makedirs(target_dir, exist_ok=True)

for filename in os.listdir(source_dir):
    if filename.lower().endswith('.jpg'):
        source_path = os.path.join(source_dir, filename)
        base_name = os.path.splitext(filename)[0]
        target_path = os.path.join(target_dir, base_name + '.bmp')

        img = Image.open(source_path).convert('RGB')
        img.save(target_path, 'BMP')
        print(f'{filename} -> {base_name}.bmp')
