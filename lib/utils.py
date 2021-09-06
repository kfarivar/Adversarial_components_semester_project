import torch
import torch.nn as nn
import logging as log

def add_normalization_layer(model, model_mean, model_std):
    ''' the attacks library requires the inputs to be in [0,1] but the model can have their own input requirements.
        Adding a normalization layer for the model. 
        We can't use torch.transforms because it supports only non-batch images.
    '''

    class Normalize(nn.Module) :
        def __init__(self, mean, std) :
            super().__init__()
            self.register_buffer('mean', torch.Tensor(mean))
            self.register_buffer('std', torch.Tensor(std))
            
        def forward(self, input):
            # Broadcasting
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
            return (input - mean) / std

    norm_layer = Normalize(mean=model_mean, std=model_std)

    new_model = nn.Sequential(
        norm_layer,
        model
    )

    return new_model





def print_measurement_results(results, measurements, on_train=True, on_val=True, on_test=True, set_log_stream=False):
    ''' Prints the results of the benchmark properly
    '''

    if set_log_stream:
        log.basicConfig(
            level=log.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                log.StreamHandler()
            ]
        )


    if on_train and results[0] is not None:
        log.info('Train set results:')
        for idx, m in enumerate(measurements):
            log.info(f'The {type(m).__name__} results:')
            log.info(results[0][idx])
        log.info('-'*20)
    if on_val and results[1] is not None:
        log.info('Validation set results:')
        for idx, m in enumerate(measurements):
            log.info(f'The {type(m).__name__} results:')
            log.info(results[1][idx])
        log.info('-'*20)
    if on_test and results[2] is not None:
        log.info('Test set results:')
        for idx, m in enumerate(measurements):
            log.info(f'The {type(m).__name__} results:')
            log.info(results[2][idx])
        log.info('-'*20)


def print_train_test_val_result(results, measurements):
    train_val_result, test_result = results

    train_measurement_results, val_measurement_results = train_val_result

    for epoch,v in train_measurement_results.items():
        log.info(f'epoch {epoch} train result:')
        print_measurement_results((v,None, None), measurements, on_val=False, on_test=False)
    
    log.info('*'*20)
    
    for epoch,v in val_measurement_results.items():
        log.info(f'epoch {epoch} validation result:')
        print_measurement_results((None,v, None), measurements, on_train=False, on_test=False)

    log.info('*'*20)

    # print the test result
    #print_measurement_results((None,None, test_result), measurements, on_val=False, on_train=False)

    #log.info('*'*20)

