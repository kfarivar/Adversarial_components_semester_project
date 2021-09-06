import torch
import torchvision
import logging as log
import torchattacks
import pickle

log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler("resnet34_batch_128_weight_decay_1e-5.log"),
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


def main(args):
    # Get the model

    if args.model=='simple':
        # simple model
        net = simple_conv_Net()
        path = './model_checkpoints/simple_conv_non_robust_cifar_epoch_10.pth'
        net.load_state_dict(torch.load(path))
        # normalization for inputs in [0,1] 
        model_mean = (0.5, 0.5 ,0.5)
        model_std = (0.5, 0.5 ,0.5)

    elif args.model=='resnet18':
        # get resnet18
        net = resnet18(pretrained=True)
        # normalization for inputs in [0,1]
        model_mean = (0.4914, 0.4822, 0.4465)
        model_std = (0.2471, 0.2435, 0.2616)
        # make untrained version
        untrained_net = resnet18()

    elif args.model == 'resnet34':
        net = resnet34(pretrained=True)
        model_mean = (0.4914, 0.4822, 0.4465)
        model_std = (0.2471, 0.2435, 0.2616)
        # make untrained version
        untrained_net = resnet34()


    # add a normalization layer
    net = add_normalization_layer(net, model_mean, model_std)

    # save state dict for the empty version of the model
    untrained_state_dict = add_normalization_layer(untrained_net, model_mean, model_std).state_dict()


    # make sure the data is in [0,1] ! if you use pytorch ToTensor tranform it is already taken care of.
    # note we have already added a normalization layer to our models to adjust them to this data.
    dataset = CIFAR10_module(mean=(0,0,0), std=(1,1,1), data_dir = "./data", batch_size=args.batch_size)
    # prepare and setup the dataset
    dataset.prepare_data()
    dataset.setup()



    # define  meaures
    normal_acc = Normal_accuracy()
    robust_acc = Robust_accuracy()

    #initialize and send the model to AdvLib
    # This has to be done before defining the attacks (and sending the model to them) 
    # otherwise the devies and the eval mode won't be set properly !!!
    # this is weird since the attacks automatically puts the model in eval mode ?!
    # This should have beem fixed by the attack library update
    model_bench = ab(net, untrained_state_dict= untrained_state_dict, device='cuda:'+args.device, predictor=lambda x: torch.max(x, 1)[1])


    model = net
    #fgsm = torchattacks.FGSM(model, eps=8/255)
    attacks = [torchattacks.FGSM(model, eps=8/255),
            #torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7),
            #torchattacks.APGD(model, eps=8/255, steps=7), # default norm inf
        ] 


    if args.mode == 'measuring':
        on_train=False
        on_val = False
        measurements = [normal_acc, robust_acc]
        results = model_bench.measure_splits(dataset, measurements, attacks, on_train=on_train, on_val=on_val)
        print_measurement_results(results, measurements, on_train=on_train)


    elif args.mode == 'robust_training':
        measurements = [normal_acc, robust_acc]

        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
        trainer = Robust_trainer(optimizer, loss)

        num_epochs = 101

        print()
        print('attacks:')
        for atk in attacks: print(type(atk).__name__)
        print()

        save_path = 'Robust_models_chpt/v3_resnet34_FGSM_batch_128_weight_decay_1e-5'
        print(save_path)
        results = model_bench.train_val_test(trainer, num_epochs, dataset, measurements, attacks, save_path,
                                            train_measure_frequency=2, val_measure_frequency=2)

        print_train_test_val_result(results, measurements)

        # save accuracy results 
        with open(save_path + '/accuracies.pkl', 'wb') as output:
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)



    elif args.mode == 'orig_robust':
        
        #original robust training used in torchattacks lib
        device = 'cuda:0'
        num_epochs = 100

        # get resnet18
        net = resnet18(pretrained=True).to(device)

        print(f"model training is: {net.training}")

        # normalization for inputs in [0,1]
        model_mean = (0.4914, 0.4822, 0.4465)
        model_std = (0.2471, 0.2435, 0.2616)

        # add a normalization layer
        net = add_normalization_layer(net, model_mean, model_std).to(device)
            

        #Attack
        fgsm = torchattacks.FGSM(net, eps=8/255)

        
        #initialise optimizer 
        """ optimizer = torch.optim.SGD(
            net.parameters(),
            lr=1e-2,
            weight_decay=1e-2,
            momentum=0.9,
            nesterov=True,
        ) """

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss = torch.nn.CrossEntropyLoss()
        
        net.train()

        from tqdm import tqdm
        train_loader = dataset.train_dataloader()
        
        # warmup didn't work better than ADAM
        """ 
        from PyTorch_CIFAR10.schduler import WarmupCosineLR
        total_steps = num_epochs * len(train_loader)
        scheduler =  WarmupCosineLR(optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps) """


        for epoch in tqdm(range(num_epochs)):
            
            for i, (batch_images, batch_labels) in tqdm(enumerate(train_loader)):
                X = fgsm(batch_images.to(device), batch_labels.to(device)).to(device)
                Y = batch_labels.to(device)

                pre = net(X)
                cost = loss(pre, Y)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                #scheduler.step()

                if (i+1) % 100 == 0:
                    print('Epoch [%d/%d], lter [%d], Loss: %.4f'
                        %(epoch+1, num_epochs, i+1, cost.item()))

        
        
        # test normal
        test_loader = dataset.test_dataloader()
        net.eval()
        correct = 0
        total = 0

        for images, labels in test_loader:
            
            images = images.to(device)
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()
            
        print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))

        #test robust
        net.eval()
        correct = 0
        total = 0

        for images, labels in test_loader:
            
            images = fgsm(images, labels).to(device)
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum()
            
        print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))








if __name__ == '__main__':
    #parse args
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument('model', type=str, choices=['simple', 'resnet18', 'resnet34'])

    parser.add_argument('mode', choices=['measuring', 'robust_training', 'orig_robust'])

    parser.add_argument('batch_size', type=int)

    parser.add_argument('device', type=str)  # e.g '2' means 'cuda:2'

    args = parser.parse_args()

    main(args)









