from PIL import Image, ImageEnhance, ImageFilter
import cv2
import skimage
import numpy as np
import os

DATASET_PATH= os.path.join("./dont_imgs")
SAVE_PATH = os.path.join("./224_augmented_imgs/")
SIZE = (224,224)

if not os.path.isdir(SAVE_PATH):
  os.mkdir(SAVE_PATH)

#Convert images to tensors, do image augmentation to it, Save it to 224x224
for label in os.listdir(DATASET_PATH):
  for img in os.listdir(os.path.join(DATASET_PATH, label)):
    raw_img = Image.open(os.path.join(DATASET_PATH, label, img))
    if not os.path.isdir(os.path.join(SAVE_PATH, label)):
      os.mkdir(os.path.join(SAVE_PATH, label))
    #Save original images
    ori_img = raw_img.resize(SIZE)
    ori_img.save(os.path.join(SAVE_PATH, label, img))
    if not ("wildebeest" or "zebra" or "gazelleThomsons") in label:
        #Horizontal Flipping
        hori_img = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
        hori_img = hori_img.resize(SIZE)
        save_name = os.path.splitext(img)[0]
        save_name = save_name + "_mirrored.JPG"
        hori_img.save(os.path.join(SAVE_PATH, label, save_name))

        #Vertical Flipping Transpose ver
        vert_img = raw_img.transpose(Image.TRANSPOSE)
        vert_img = vert_img.resize(SIZE)
        save_name = os.path.splitext(img)[0]
        save_name = save_name + "_vertical_transpose.JPG"
        vert_img.save(os.path.join(SAVE_PATH, label, save_name))

        #Vertical Flipping Bottom ver
        vert_img = raw_img.transpose(Image.FLIP_TOP_BOTTOM)
        vert_img = vert_img.resize(SIZE)
        save_name = os.path.splitext(img)[0]
        save_name = save_name + "_vertical.JPG"
        vert_img.save(os.path.join(SAVE_PATH, label, save_name))

        #Low Brightness Modification
        #enhance_img = ImageEnhance.Brightness(raw_img)
        #bright_img = enhance_img.enhance(0.4).resize(SIZE)
        #save_name = os.path.splitext(img)[0]
        #save_name = save_name + "_low_brightness.JPG"
        #bright_img.save(os.path.join(SAVE_PATH, label, save_name))

        #Contrast Modification
        enhance_img = ImageEnhance.Contrast(raw_img)
        const_img = enhance_img.enhance(0.4).resize(SIZE)
        save_name = os.path.splitext(img)[0]
        save_name = save_name + "_low_constrast.JPG"
        const_img.save(os.path.join(SAVE_PATH, label, save_name))

        #Add gaussian noise to the image
        image = skimage.io.imread(os.path.join(DATASET_PATH, label, img))
        image = skimage.transform.resize(image, SIZE)
        image = 255 * image
        image = image.astype(np.uint8)
        gauss_img = skimage.util.random_noise(image, mode='gaussian')
        save_name = os.path.splitext(img)[0]
        save_name = save_name + "_gaussian_noise.JPG"
        skimage.io.imsave(os.path.join(SAVE_PATH, label, save_name), gauss_img)