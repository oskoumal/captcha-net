import os
import glob
import cv2
from pathlib import Path

from definitions import ROOT_DIR
from preprocessing.filters import hard_noise_removing_filter

CAPTCHA_IMAGE_PATH = Path("data/captcha_training_images")
OUTPUT_PATH = Path("data/captcha_training_images_preprocessed")

# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(ROOT_DIR, CAPTCHA_IMAGE_PATH, "*"))
counts = {}
num_of_characters_detected = 0
num_of_correct_preprocess = 0

print("Number of images:", len(captcha_image_files))
show_preprocess = False

# Loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))
    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Read image
    image = cv2.imread(captcha_image_file)

    # Clean up the image by removing noise
    image = hard_noise_removing_filter(image)

    cv2.imwrite(os.path.join(ROOT_DIR, OUTPUT_PATH, filename), image)