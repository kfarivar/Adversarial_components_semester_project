import torch
import torchvision
import logging as log
import torchattacks
import pickle
import numpy as np

log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler("resenet_SGD_warmup_debug.log"),
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



from art.metrics import clever_u
from art.estimators.classification import PyTorchClassifier

device = "cuda:0"

# get resnet18
net = resnet18(pretrained=True)
# normalization for inputs in [0,1]
model_mean = (0.4914, 0.4822, 0.4465)
model_std = (0.2471, 0.2435, 0.2616)
# add a normalization layer
net = add_normalization_layer(net, model_mean, model_std).to(device)
# load into framework
classifier = PyTorchClassifier(
    model=net,
    loss=torch.nn.CrossEntropyLoss(),
    input_shape=[3, 32, 32],
    nb_classes=10,
)

# load the robust network
robust_net = resnet18(pretrained=False)
# normalization for inputs in [0,1]
model_mean = (0.4914, 0.4822, 0.4465)
model_std = (0.2471, 0.2435, 0.2616)
robust_net = add_normalization_layer(robust_net, model_mean, model_std).to(device)
# load model
path = 'Robust_models_chpt/v2_resnet18_FGSM/epoch_45.pt'
robust_net.load_state_dict(torch.load(path))
robust_net.eval()
# load to framework
robust_classifier = PyTorchClassifier(
    model=robust_net,
    loss=torch.nn.CrossEntropyLoss(),
    input_shape=[3, 32, 32],
    nb_classes=10,
)

dataset = CIFAR10_module(mean=(0,0,0), std=(1,1,1), data_dir = "./data")
# prepare and setup the dataset
dataset.prepare_data()
dataset.setup()

#model_bench = ab(net, untrained_state_dict= untrained_state_dict, device='cuda:1', predictor=lambda x: torch.max(x, 1)[1])

from torchvision.utils import save_image
import torchvision.transforms as transforms

for b_idx, data in enumerate(dataset.test_dataloader()):
    if b_idx==1:
        inputs = data[0] #.to(device)
        indexes = data[1]

        

        #print(list(inputs.shape[1:]))

        #print( net(inputs.to(device)).shape[1] )

        print(indexes)
        
        current_indexes = np.intersect1d([1,4,7], indexes)

        print(current_indexes)

        """ save_image(inputs[0], 'images/img_orig.png')

        data_transforms = torch.nn.Sequential(
        transforms.RandomCrop(32, padding=4)
        )
        for i in range(0, 5):
            save_image( data_transforms(inputs[0]), f'images/v{i}.png'  ) """

        """
        print(example)

        print()
        print(np.array( [example] )  ) """

        scores = []
        rob_scores = []

        for data in inputs:
            example = data.numpy()

            score = clever_u(classifier=classifier , x=example , nb_batches=100, batch_size=64, radius=8/255, norm=np.inf, c_init=1.0, pool_factor=2, verbose=True)
            print("clever score normal: " , score)

            rob_score = clever_u(classifier=robust_classifier , x=example , nb_batches=100, batch_size=64, radius=8/255, norm=np.inf, c_init=1.0, pool_factor=2, verbose=True)
            print("robust clever score: ", rob_score)

            scores.append(score)
            rob_scores.append(rob_score)

        print(scores)
        print(rob_scores)

        print("avg score:" , np.mean(scores))
        print("avg rob score: ", np.mean(rob_scores))