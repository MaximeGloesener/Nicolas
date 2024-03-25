# imports 
import torch 
from torchvision.transforms import *
from registry import get_model
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'cifar100': dict( mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2761)),
    }
image_size = 32
transforms = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(**NORMALIZE_DICT["cifar10"]),
    ]),
    "test": Compose([
        ToTensor(),
        Normalize(**NORMALIZE_DICT["cifar10"]),
    ]),
}

img_path = "dog.jpg"
test_image = Image.open(img_path).convert("RGB")
input_tensor = transforms["test"](test_image).unsqueeze(0).to(device)
test_image = np.array(test_image)
test_image = cv2.resize(test_image, (32, 32))
test_image = test_image.astype(np.float32)
test_image = test_image / 255.0
imgplot = plt.imshow(test_image)
plt.show()

# définition du modèle
model = get_model("resnet56", "cifar10").to(device)
target_layers = [model.layer3[7].conv2]

# méthodes existantes: GradCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM, EigenCAM, FullGrad, AblationCAM, HiResCAM
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor)
grayscale_cam = grayscale_cam[0, :]
# visualisation de la heat map
visualization = show_cam_on_image(test_image, grayscale_cam, use_rgb=True)
model_outputs = cam.outputs
imgplot = plt.imshow(visualization)
plt.show()

