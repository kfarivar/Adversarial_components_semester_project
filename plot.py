import torch
import torchvision
import logging as log
import torchattacks
import pickle

log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler("v2_resnet34.log"),
        log.StreamHandler()
    ]
)

from lib.AdvLib import Adversarisal_bench as ab
from lib.simple_model import simple_conv_Net
from lib.Get_dataset import CIFAR10_module
from lib.Measurements import Normal_accuracy, Robust_accuracy
from lib.utils import print_measurement_results, print_train_test_val_result, add_normalization_layer
from lib.Trainer import Robust_trainer

from PyTorch_CIFAR10.cifar10_models.resnet import resnet18 , resnet34





normal_acc = Normal_accuracy()
robust_acc = Robust_accuracy()
measurements = [normal_acc, robust_acc]


with open('Robust_models_chpt/v2_resnet34_FGSM_lr-4_batch_64/accuracies.pkl', 'rb') as handle:
    results = pickle.load(handle)


print_train_test_val_result(results, measurements)
""" 
train_val_result, _ = results

train_measurement_results, val_measurement_results = train_val_result """