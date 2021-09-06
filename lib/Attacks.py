import torch
import torch.nn as nn
import torch.tensor as tensor

from abc import ABC, abstractmethod

class Attack(ABC):
    '''abstract class for different types of attacks to inherit from'''

    @abstractmethod
    def __init__(self, model:nn.Module, loss:nn.Module):
        self.model = model
        self.loss =loss

    @abstractmethod
    def attack(sample, label, target)->(tensor, int):
        '''assumes the prediction of the model on this sample is correct.
        if attack succeeds should return the tuple: (distorted input, label) 
        if fail should return None'''
        assert self.model(sample) == label, "the classification is already incorrect !"
        


class FGSM(Attack):

    def __init__(self, model:nn.Module, loss:nn.Module, epsilon):
        super().__init__(model, loss)
        self.epsilon = epsilon

    
    def attack(sample, label):
        super().attack(sample, label)

        # keep track of gradient
        sample.requires_grad = True

        # predict
        output = self.model(sample)

        # Calculate the loss
        loss = self.loss(output, label)

        # Zero all existing gradients
        self.model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = sample.grad.data

        # perform FGSM Attack
        perturbed_data = _fgsm_attack(sample, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output

        if final_pred.item() == label.item():
            # the attack was unsuccessful
            return None
        else:
            # send back the results if successful
            return (perturbed_data, final_pred.item())

    
    def _fgsm_attack(image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image

