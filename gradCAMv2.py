# imports 
import torch
import torch.nn as nn 
from torchvision.transforms import *
from registry import get_model
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

import os 
import re 

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

# définition du modèle
model_name = "resnet56"
model = get_model(model_name, "cifar10").to(device)
target_layers = [model.layer3[7].conv2]
# méthodes existantes: GradCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM, EigenCAM, FullGrad, AblationCAM, HiResCAM
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor)
grayscale_cam = grayscale_cam[0, :]
# visualisation de la heat map
visualization = show_cam_on_image(test_image, grayscale_cam, use_rgb=True)
model_outputs = cam.outputs
imgplot = plt.imshow(visualization)
plt.axis('off')
plt.tight_layout()
# plt.show()


# model = torch.load('results\\resnet56_compression_2.0_cifar10_l1.pth').to(device)
# print(model.eval())


path = "images"
methods = ['group_norm','group_sl','l1','lamp','magnitude_local','random','slim']
pruned_path = 'results'
for pruned_model in os.listdir(pruned_path):
    method = re.search(r'cifar10_(.*?)\.pth', pruned_model).group(1)       
    model_name = re.search(r'(.*?)_compression', pruned_model).group(1)
    compression_ratio = float(re.search(r'compression_(.*?)_', pruned_model).group(1))
    print(f'** Modèle = {model_name} **')
    print(f'** Méthode = {method} **')
    print(f'** Compression Ratio = {compression_ratio} **')
    model = torch.load(os.path.join(pruned_path, pruned_model)).to(device)
    print(model.eval()) 

    # target layer = last conv layer
    last_conv_layer = None 
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv_layer = module
    target_layers =  [last_conv_layer]

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    # visualisation de la heat map
    visualization = show_cam_on_image(test_image, grayscale_cam, use_rgb=True)
    model_outputs = cam.outputs
    imgplot = plt.imshow(visualization)
    plt.savefig(f'{path}/dog_{method}_{compression_ratio}_{model_name}.png')
    plt.title(f'{method} - {compression_ratio} - {model_name}', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    #plt.show()
    print("\n\n")
 
# évaluation de l'explication
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.sobel_cam import sobel_cam


targets = [ClassifierOutputSoftmaxTarget(6)]
cam_metric = CamMultImageConfidenceChange()
inverse_cams = 1 - grayscale_cam
sobel_cam_grayscale = sobel_cam(np.uint8(test_image * 255))
thresholded_cam = sobel_cam_grayscale < np.percentile(sobel_cam_grayscale, 75)
grayscale_cam = cam(input_tensor=input_tensor)
thresholded_cam = grayscale_cam < np.percentile(grayscale_cam, 75)
scores, visualizations = cam_metric(input_tensor, grayscale_cam, targets, model, return_visualization=True)
#scores, visualizations = cam_metric(input_tensor, thresholded_cam, targets, model, return_visualization=True)
#scores, visualizations = cam_metric(input_tensor, inverse_cam, targets, model, return_visualization=True)
#scores, visualizations = cam_metric(input_tensor, sobel_cam, targets, model, return_visualization=True)
score = scores[0]
visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
visualization = deprocess_image(visualization)
print(f"The confidence increase percent: {100*score}")
print("The visualization of the pertubated image for the metric:")

Image.fromarray(visualization).show()

from pytorch_grad_cam.metrics.cam_mult_image import DropInConfidence, IncreaseInConfidence

print("Drop in confidence", DropInConfidence()(input_tensor, grayscale_cam, targets, model))
print("Increase in confidence", IncreaseInConfidence()(input_tensor, grayscale_cam, targets, model))