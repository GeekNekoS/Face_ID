import zipfile
import albumentations as A
import csv
import cv2
from random import random


max_count = 202599
transform1 = A.Compose(
    [A.CLAHE(),
     A.RandomRotate90(always_apply=True),
     A.ChannelShuffle(),
     A.CoarseDropout(),
     A.Blur(blur_limit=3)]
)
transform2 = A.Compose(
    [A.Sharpen(),
     A.ColorJitter((0.8, 1), (0.8, 1)),
     A.Flip(),
     A.GaussNoise(var_limit=70, mean=10, always_apply=True)]
)
transform3 = A.Compose(
    [A.RandomToneCurve(scale=0.8, always_apply=True),
     A.RandomShadow(always_apply=True)]
)
transform4 = A.Compose(
    [A.CLAHE(),
     A.RandomRotate90(),
     A.Transpose(),
     A.Blur(blur_limit=3),
     A.HueSaturationValue()])
test_dataset_images_path = '../datasets/data'
new_dataset_path = '../datasets/faces_and_identifiers.csv'
test_dataset_identifiers_path = '../datasets/identity_CelebA.txt'
counter = 0
with zipfile.ZipFile(test_dataset_images_path, 'r'):
    with open(test_dataset_identifiers_path, 'a') as identifiers:
        with open(new_dataset_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                row = dict(row)
                identifier = row['identifier']
                pictures_str = row['pictures']
                pictures_lst = []
                picture_path = ''
                for element in pictures_str:
                    if element not in "[]', ":
                        picture_path += element
                    else:
                        if picture_path:
                            pictures_lst.append(picture_path)
                            picture_path = ''
                count = ((200 // len(pictures_lst)) + 1) if 200 % len(pictures_lst) != 0 else 200 // len(pictures_lst)
                for path in pictures_lst:
                    image_path = test_dataset_images_path + '/' + path
                    image = cv2.cvtColor(cv2.imread(image_path),  cv2.COLOR_BGR2RGB)
                    for _ in range(count):
                        counter += 1
                        max_count += 1
                        chance = random()
                        if chance <= 0.25:
                            augmented_image = transform1(image=image)['image']
                        elif 0.25 < chance <= 0.5:
                            augmented_image = transform2(image=image)['image']
                        elif 0.5 < chance <= 0.75:
                            augmented_image = transform3(image=image)['image']
                        else:
                            augmented_image = transform4(image=image)['image']
                        print(counter)
