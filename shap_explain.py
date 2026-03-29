import shap
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("alzheimers_model.h5")

img = image.load_img("test_image.jpg", target_size=(96,96))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

background = np.random.rand(10,96,96,3)

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(img_array)

shap.image_plot(shap_values, img_array)