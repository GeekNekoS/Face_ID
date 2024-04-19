import os
import cv2
import keras
import keras_cv
import numpy as np
from PIL import Image
from keras_cv import visualization
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "tensorflow"

pretrained_model = keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

filepath = "data/nekos_images/test_8.jpg"
# filepath_1 = "data/collected_images/6b081f20-ee5f-11ee-98be-f42679e0f8f2.jpg"

image = keras.utils.load_img(
    filepath,
    color_mode="rgb",
    target_size=(224, 224),
    interpolation="nearest",
    keep_aspect_ratio=False,
)

# image = cv2.imread(filepath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

inference_resizing = keras_cv.layers.Resizing(
    224, 224, pad_to_aspect_ratio=True, bounding_box_format="xywh"
)
image_batch = inference_resizing([image])

# image_batch = tf.data.Dataset.from_tensor_slices([image])

class_ids = [
    "Face",
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

prediction_decoder = keras_cv.layers.NonMaxSuppression(
    bounding_box_format="xywh",
    from_logits=True,
    iou_threshold=0.2,
    confidence_threshold=0.7,
)

y_pred = pretrained_model.predict(image_batch)
output_image_pil = visualization.plot_bounding_box_gallery(
    image_batch,
    value_range=(0, 255),
    rows=1,
    cols=1,
    y_pred=y_pred,
    scale=2,
    font_scale=0.7,
    bounding_box_format="xywh",
    class_mapping=class_mapping,
)

output_image_pil.savefig("output_image.png")
output_image_pil.close()

output_image_pil = Image.open("output_image.png")
output_image_array = np.asarray(output_image_pil, dtype=np.uint8)
output_image = cv2.cvtColor(output_image_array, cv2.COLOR_RGB2BGR)

cv2.imshow('Detection Results', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

input_arr = keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = pretrained_model.predict(input_arr)
