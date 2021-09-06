import torch
import torchvision
import torchattacks
import pickle
import logging as log

log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler("checkpoint_evaluation.log"),
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


import glob
import re

##
# Change: 1. path 2. model 3. attack 4. train/val flags

# For resnet 34 epoch 15 seems the best


# measure checkpoints
path = 'Robust_models_chpt/resnet34_FGSM'
# list of chekpoints
chpts_list = sorted(glob.glob(path+'/*.pt'), key=lambda x: int(re.findall(r'\d+.pt', x)[0][:-3]) )

dataset = CIFAR10_module(mean=(0,0,0), std=(1,1,1), data_dir = "./data")
# prepare and setup the dataset
dataset.prepare_data()
dataset.setup()

print(chpts_list)

for chpt in chpts_list:
    #get epoch
    epoch = re.findall(r'\d+.pt', chpt)[0][:-3]
    print(f'\nMeasuring epoch {epoch}:')  

    # first add the norm layer then load
    model = resnet34()
    model_mean = (0.4914, 0.4822, 0.4465)
    model_std = (0.2471, 0.2435, 0.2616)
    # add a normalization layer
    net = add_normalization_layer(model, model_mean, model_std)

    net.load_state_dict(torch.load(chpt))
    net.eval()


    # define  meaures
    normal_acc = Normal_accuracy()
    robust_acc = Robust_accuracy()

    #initialize and send the model to AdvLib
    model_bench = ab(net, untrained_state_dict= None, device='cuda:1', predictor=lambda x: torch.max(x, 1)[1])

    model = net
    attacks = [
            torchattacks.FGSM(model, eps=8/255),
            #torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=7),
            #torchattacks.CW(model, c=1, kappa=0, steps=1000, lr=0.01),
            #torchattacks.RFGSM(model, eps=8/255, alpha=4/255, steps=1),
            #torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7),
            #torchattacks.FFGSM(model, eps=8/255, alpha=12/255),
            #torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=7),
            #torchattacks.MIFGSM(model, eps=8/255, decay=1.0, steps=5),
            #torchattacks.APGD(model, eps=8/255, steps=7), # default norm inf
            #torchattacks.FAB(model, eps=8/255),
            #torchattacks.Square(model, eps=8/255),
            #torchattacks.PGDDLR(model, eps=8/255, alpha=2/255, steps=7),
        ] 

    # only measure on test
    on_train=True
    on_val = False
    measurements = [normal_acc, robust_acc]
    results = model_bench.measure_splits(dataset, measurements, attacks, on_train=on_train, on_val=on_val)

    print_measurement_results(results, measurements, on_train=on_train, set_log_stream=True) 

