import torch
import torchvision
import logging as log
import torchattacks
import pickle
import numpy as np

""" log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler("resenet_SGD_warmup_debug.log"),
        log.StreamHandler()
    ]
) """

from lib.AdvLib import Adversarisal_bench as ab
from lib.simple_model import simple_conv_Net
from lib.Get_dataset import CIFAR10_module
from lib.Measurements import Normal_accuracy, Robust_accuracy
from lib.utils import print_measurement_results, print_train_test_val_result, add_normalization_layer
from lib.Trainer import Robust_trainer

from PyTorch_CIFAR10.cifar10_models.resnet import resnet18 , resnet34


device = "cuda:0"

# get resnet18
net = resnet18(pretrained=True)
# normalization for inputs in [0,1]
model_mean = (0.4914, 0.4822, 0.4465)
model_std = (0.2471, 0.2435, 0.2616)
# add a normalization layer
net = add_normalization_layer(net, model_mean, model_std).to(device)
net.eval()

# load the robust network
robust_net = resnet18(pretrained=False)
# normalization for inputs in [0,1]
model_mean = (0.4914, 0.4822, 0.4465)
model_std = (0.2471, 0.2435, 0.2616)
robust_net = add_normalization_layer(robust_net, model_mean, model_std).to(device)
# load model
path = 'Robust_models_chpt/v2_resnet18_FGSM/epoch_45.pt'
#path = 'Robust_models_chpt/v2_resnet34_FGSM_lr-4_batch_64/epoch_100.pt'


robust_net.load_state_dict(torch.load(path))
robust_net.eval()

dataset = CIFAR10_module(mean=(0,0,0), std=(1,1,1), data_dir = "./data")
# prepare and setup the dataset
dataset.prepare_data()
dataset.setup()

from torchvision.utils import save_image
import matplotlib.pyplot as plt

for b_idx, data in enumerate(dataset.test_dataloader()):
    if b_idx==1:
        inputs = data[0].to(device)
        labels = data[1].to(device)
        indexes = data[2]
        print(indexes)

        inputs.requires_grad = True
        # get saliency maps of both models

        # calc grads wrt input
        # the autograd only supports scalars as inputs (can't do batch)
        # also we have to do this one sample at a time. 
        # can't take gradient wrt single sample if the forward is batch 
        for idx, single_in in enumerate(inputs):
            # forward
            single_out = net( torch.unsqueeze(single_in, 0) )

            print("sing out shape: ", single_out.shape)

            
            
            loss = torch.nn.CrossEntropyLoss()
            cost = loss(single_out, torch.unsqueeze(labels[idx], 0))

            grad = torch.autograd.grad(cost, single_in, retain_graph=False, create_graph=False)[0]
            print("grad shape",grad.shape)



            """# get the max logit
            max_logit= torch.max(single_out)

            grad = torch.autograd.grad(max_logit, single_in, retain_graph=False, create_graph=False)[0]
            print("grad shape",grad.shape) """
           
            '''
            Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
            R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
            across all colour channels.'''
            
            #saliency, _ = torch.max(grad.data.abs(),dim=0)
            saliency = torch.mean(grad.data.abs(),dim=0)
            print("saliency shape: " , saliency.shape)

            # clamp and scale
            """ mean = torch.mean(saliency)
            std = torch.std(saliency)
            clamped_sal = torch.clamp(saliency-mean, min= -3*std, max=3*std)

            center = (clamped_sal-torch.mean(clamped_sal))  
            scaled_sal = center / (torch.max(center) - torch.min(center) )
            scaled_sal = scaled_sal - torch.min(scaled_sal) """

            # save the saliency map as a heatmap
            name = 'mean'
            plt.imsave(f'images/saliencies/{name}_sal_{indexes[idx]}.png', saliency.cpu(), cmap=plt.cm.hot)
            #  save original image
            save_image(single_in, f'images/orig/orig_{indexes[idx]}.png')
        
            

        #################
        #Do the robust model now (or make a function)


            