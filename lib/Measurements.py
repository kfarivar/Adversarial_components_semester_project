import numpy as np
import torch
import torch.nn as nn
import attr
from abc import ABC, abstractmethod

from art.metrics import clever_u
from art.estimators.classification import PyTorchClassifier

class Measure(ABC):
    ''' A specific measurement to be applied on a batch of data in measure method of AdvLib.
    '''
    
    @abstractmethod
    def on_clean_data():
        '''called on the result of a batch of data.
        '''
        pass

    @abstractmethod
    def on_attack_data():
        ''' called on the result of an attack on a batch of data.
        '''
        pass

    @abstractmethod
    def batch_end():
        '''called after calcuating each single batch.
        '''
        pass

    @abstractmethod
    def final_result():
        '''called after the whole dataset is processed.
        '''
        pass




@attr.s
class Normal_accuracy(Measure):
    ''' Measure a specific models non-robust / normal accuracy.
    '''
    # total number of data points
    _total = attr.ib(default=0)
    # number of corrcet predictions
    _correct = attr.ib(default=0) 

    def on_clean_data(self, model, inputs, labels, outputs, predicted_labels, indexes):
        self._total += labels.size(0)
        self._correct += (predicted_labels == labels).sum().item()


    def on_attack_data(self,*args):
        return None

    def batch_end(self):
        return None
        
    def final_result(self):
        accuracy = self._correct/self._total

        ############################
        # IMPORTANT
        #reset the values
        self._total = 0
        self._correct = 0

        return accuracy *100
        


@attr.s
class Robust_accuracy(Measure):
    ''' Measures the accuracy of the model with respect to a certain attack 
    '''
    # total number of data points
    _total = attr.ib(default=0)
    # total number of data points that are correctly predicted Before applying the attack
    _total_correct = attr.ib(default=0)


    # number of data points that were correctly predicted after attack (regadrless of, if they were initially predicted correctly or not)
    # the keys are the attack names type(attack).__name__
    _correct_after_attack = attr.ib(factory=dict)
    
    # number of data points that were corrcetly predicted both before and after applying the attack
    _correct_before_and_after_attack = attr.ib(factory=dict) 

    def on_clean_data(self, model, inputs, labels, outputs, predicted_labels, indexes):
        self._total += labels.size(0)
        self._total_correct += (predicted_labels == labels).sum().item()



    def on_attack_data(self, model, inputs, labels, outputs, predicted_labels, adv_inputs, adv_output, adv_predictions, attack, indexes):
        ''' Takes a batch and the adverserial version of the batch and returns what precent of the previously correct are correct after the
            attack and what percent of total are now correct.
        '''
        # separate sampels that are correctly classified
        mask_correct = (labels == predicted_labels)
        correctly_predicted_labels = labels[mask_correct]
        corresponding_adv_predictions = adv_predictions[mask_correct]
        attack_name = type(attack).__name__
        # count the total correct after this attack
        self._correct_after_attack[attack_name] = self._correct_after_attack.get(attack_name, 0) + (labels == adv_predictions).sum().item()
        # count the total that were correct before and after the attack
        self._correct_before_and_after_attack[attack_name] = self._correct_before_and_after_attack.get(attack_name,0) + \
                                                            (correctly_predicted_labels == corresponding_adv_predictions).sum().item()

    
    def batch_end(self):
        return None
        
    def final_result(self):
        results = {}
        for attack_name, correct_after_attack in self._correct_after_attack.items():
            correct_before_and_after_attack = self._correct_before_and_after_attack[attack_name]
            #initialise a sub dict for this attack's result
            results[attack_name] = {}
            # return the two stats
            results[attack_name]['percentage_correct_after_attack'] = correct_after_attack / self._total * 100
            results[attack_name]['percentage_of_correct_that_remain_correct_after_attack'] = correct_before_and_after_attack / self._total_correct * 100

        # reset the counters
        self._total = 0
        self._total_correct = 0
        self._correct_after_attack = {}
        self._correct_before_and_after_attack = {}

        return results


@attr.s
class Subset_saliency(Measure):
    pass



# needs debuging
@attr.s
class Clever_score(Measure):
    ''' Note, the implemnetation of the adversarial-robustness-toolbox is inefficient and incomplete !  
    Clever is an estimate of the minimum amount of purturbation required to create an adverserial example for a specific data point.
    This measure returns the average score for a subset of the data set. 
    see paper: https://arxiv.org/abs/1801.10578
    The adversarial-robustness-toolbox implementation is used. (https://github.com/Trusted-AI/adversarial-robustness-toolbox)
    
    The number of classes should match the second dimension of outputs.  
    '''
    #a set of indexes for the subset of points to calculate the score on
    # array-like (e.g. list)
    data_point_indexes = attr.ib()
    #The loss function used for training (e.g. for classification torch.nn.CrossEntropyLoss())
    loss = attr.ib()
    # refer to library docs and paper
    nb_batches:int=attr.ib()
    batch_size:int=attr.ib()
    # radius similar to the epsilon in an attack
    radius:float = attr.ib()
    # either 1,2,np.inf
    norm = attr.ib()
    c_init = attr.ib()
    pool_factor = attr.ib()
    verbose = attr.ib()


    #(internal) dict of the clever scores. index is key clever score is value.
    _clever_scores = attr.ib(default=dict()) 

    def on_clean_data(self, model, inputs, labels, outputs, predicted_labels, indexes):
        # load the model into the librray format
        classifier = PyTorchClassifier(
                                    model=model,
                                    loss=self.loss,
                                    input_shape=list(inputs.shape[1:]),
                                    nb_classes=outputs.shape[1],
                                ) 

        # get indexes we are interested in.
        # access_indexes is the index we can use to get the desired input
        data_indexes, _, access_indexes = np.intersect1d(self.data_point_indexes, indexes)
        # this for loop can be parallelized in cpu
        for data_idx, access_idx in zip(data_indexes,access_indexes):
            # data points can be duplicates avoid double calc
            if data_idx not in self._clever_scores:
                # get the access index
                data_point = inputs[access_idx].numpy() 

                clever_score = clever_u(classifier=classifier , x=data_point, nb_batches=self.nb_batches, batch_size=self.batch_size, 
                                radius=self.radius, norm=self.norm, c_init=self.c_init, pool_factor=self.pool_factor, verbose=self.verbose)

                self._clever_scores[data_idx] = clever_score



    def on_attack_data(self,*args):
        return None

    def batch_end(self):
        return None
        
    def final_result(self):
        avg_clever_score = np.mean(list(self._clever_scores.values()))
        ############################
        # IMPORTANT
        #reset the values
        self._clever_scores = dict()

        return avg_clever_score
    



class Dataset_measure(ABC):
    ''' for measures specific to the dataset. 
        These usually require the whole dataset to calculate things like knn or clustering.
    '''
    @abstractmethod
    def on_data():
        pass

        





        
