import torch
import cv2
import numpy as np
import supervision as sv


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

from skimage.io import imread


IMAGE_PATH = "coca.jpeg"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
# Cargar la imagen
image = imread(IMAGE_PATH)

# Crear un objeto Segmenter
segmenter = SamPredictor(sam)

# Segmentar la imagen en regiones
regions = segmenter.set_image(image)

# Mostrar la imagen original y las regiones segmentadas
import matplotlib.pyplot as plt
fig, ax = plt.subplots(ncols=2, figsize=(10,5))
ax[0].imshow(image)
ax[0].set_title("Imagen original")
ax[1].imshow(regions)
ax[1].set_title("Regiones segmentadas")
plt.show()