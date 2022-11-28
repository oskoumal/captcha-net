import os

import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from pathlib import Path

from tqdm import tqdm

from definitions import ROOT_DIR, MODEL_PATH
from training.custom_image_dataset_from_directory import CustomImageDatasetFromDirectory
from training.captcha_coder import Coding, CaptchaCoder
from training.models.resnet_101_mse import ResNet101Mse
from training.models.resnet_152_mse import ResNet152Mse
from training.models.resnet_50_mse import ResNet50Mse
from training.models.resnext_101_32x8d_mse import ResNext10132x8d
from training.utils.training_utils import get_default_device, DeviceDataLoader, to_device, evaluate, fit

CAPTCHA_IMAGE_PATH = ROOT_DIR / Path("training_data/captcha_training_images")

# Problem parameters
ALPHABET_SIZE = 33
CAPTCHA_CHARACTERS = 5

# Model hyperparameters
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.001
#OPT_FUNC = torch.optim.Adam
OPT_FUNC = torch.optim.SGD

# Prepare dataset and set up pipeline
transformations = torchvision.transforms.Compose([
    # Converts the PIL image with a pixel range of [0, 255] to a PyTorch FloatTensor of shape (C, H, W) with a range
    # [0.0, 1.0]
    torchvision.transforms.ToTensor(),
    # This normalizes the tensor image with mean and standard deviation
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Resize image to 224 x 224 as required by most vision models
    torchvision.transforms.Resize(size=(224, 224))
])


def train():
    dataset = CustomImageDatasetFromDirectory(str(CAPTCHA_IMAGE_PATH), encoding=Coding.full_one_hot,
                                              transform=transformations)
    print("Number of CAPTCHAs in training set: ", dataset.num_of_samples)

    # 7149
    val_size = 1191
    train_size = dataset.num_of_samples - val_size

    print("Num of classes:", dataset.get_num_of_classes())
    print("Class names:", dataset.get_class_names())
    print("Class occurrences:", dataset.get_labels_occurrences_map())

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print("Train set size:", len(train_ds))
    print("Validation set size:", len(val_ds))

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, num_workers=0)

    device = get_default_device()
    print("Actual running device:", device)

    train_loader_cuda = DeviceDataLoader(train_loader, device)
    val_loader_cuda = DeviceDataLoader(val_loader, device)

    # Verify dataset
    for images, _ in train_loader:
        print('CAPTCHA images.shape:', images.shape)
        plt.axis('off')
        plt.imshow(images[0].permute(1, 2, 0))
        break
    plt.show()

    # Define model
    #model = ResNet50Mse(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    #model = ResNet152Mse(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    model = ResNext10132x8d(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    to_device(model, device)

    # Verify model output
    for images, _ in train_loader_cuda:
        out = model(images[0].view(1, 3, 224, 224))
        print(out.size())
        break

    # Train model
    history = [evaluate(model, val_loader_cuda)]
    print(history)
    history = fit(EPOCHS, LR, model, train_loader_cuda, val_loader_cuda, OPT_FUNC)
    # TODO - Train model - Transfer learning


def test():
    dataset = CustomImageDatasetFromDirectory(str(ROOT_DIR / Path("training_data/captchy_do_01_08_2022")),
                                              encoding=Coding.full_one_hot, transform=transformations)
    test_ds = dataset
    print("Number of CAPTCHAs in test set: ", dataset.num_of_samples)

    # Only 1 because of completely evaluating every single test sample
    batch_size = 1
    test_loader = DataLoader(test_ds, batch_size, num_workers=0)

    device = get_default_device()
    print("Actual running device:", device)

    test_loader_cuda = DeviceDataLoader(test_loader, device)

    # Verify dataset
    for images, _ in test_loader:
        print('CAPTCHA images.shape:', images.shape)
        plt.axis('off')
        plt.imshow(images[0].permute(1, 2, 0))
        break
    plt.show()

    # Define model
    model_name = "ResNet101Mse_acc=0.9876644611358643.pt"

    model = ResNet101Mse(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    model.load_state_dict(torch.load(MODEL_PATH / model_name))
    print("Model", model_name, "have been loaded")
    model.eval()

    to_device(model, device)

    correct = 0
    error_str = ""
    coder = CaptchaCoder(Coding.full_one_hot)
    with torch.no_grad():
        for batch in tqdm(test_loader_cuda):
            images, labels = batch

            # Generate predictions
            out = model(images)

            # Compute fully decoded CAPTCHA words, for easy debugging
            prediction = coder.decode_raw_output(out.cpu())
            ground_truth = coder.decode(labels.cpu())
            if prediction == ground_truth:
                correct += 1
            else:
                print("Mistake, predicted:", prediction, " ,but correct is:", ground_truth)
            # print("prediction:", prediction)
            # print("ground truth:", ground_truth)

    wrong = dataset.num_of_samples - correct
    precision = correct / dataset.num_of_samples
    print("Correct:", correct, "Wrong:", wrong)
    print("Precision:", precision)

