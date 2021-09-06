''' Run it to train a simple non-robust model on CIFAR10
'''
import torch
import torch.nn as nn
import torch.optim as optim

from Get_dataset import CIFAR10_module
from simple_model import Net

def non_robust_training(net:nn.Module, trainloader, testloader, epochs, device):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # save the model every other epoch
        if epoch%2==0:
            PATH = f'./non_robust_model_checkpoints/simple_conv_non_robust_cifar_epoch_{epoch}.pth'
            torch.save(net.state_dict(), PATH)
            # test the model
            acc = test_acc(model, testloader)
            print(f'Accuracy of the network on the 10000 test images in  epoch:{epoch} is : {100 * acc}')

    print('Finished Training')
    return  net

def test_acc(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


if __name__ == "__main__":
    #get data
    dataset = CIFAR10_module()
    # prepare and setup the dataset
    dataset.prepare_data()
    dataset.setup()
    trainloader = dataset.train_dataloader()
    testloader = dataset.test_dataloader()

    #use GPU
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print('Device is:' , device)
    test = True

    if test:
        # get the simple model
        net = Net()
        path = './non_robust_model_checkpoints/simple_conv_non_robust_cifar_epoch_10.pth'
        net.load_state_dict(torch.load(path))
        net.to(device)
        acc = test_acc(net, trainloader)
        print(f'accuracy is {acc}')



    else: # train
        # get model
        model = Net().to(device)
        #train the model normally (non_robust) and save it
        model = non_robust_training (model, trainloader, testloader, epochs=20, device=device)
    
    

     