import math
import pickle
import torch
import torch.cuda
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as dsets
import os
from utils.BBBConvmodel import BBBAlexNet, BBBLeNet
from utils.BBBlayers import GaussianVariationalInference

cuda = torch.cuda.is_available()

'''
HYPERPARAMETERS
'''
save_model = True
is_training = True  # set to "False" for evaluation of network ability to remember previous tasks
pretrained = False  # change pretrained to "True" for continual learning

if pretrained is False:
    task = 1
    noise = 0
elif pretrained is True:
    task = 2  # change to 3, 4, 5, etc. for more tasks
    noise = 0.25  # add extent of Gaussian noise

num_samples = 10  # because of Casper's trick
batch_size = 32
beta_type = "Blundell"
net = BBBLeNet   # LeNet or AlexNet
dataset = 'MNIST'  # MNIST or CIFAR-100
num_epochs = 100
p_logvar_init = 0
q_logvar_init = -10
lr = 0.00001
weight_decay = 0.0005


# dimensions of input and output
if dataset is 'MNIST':    # train with MNIST
    outputs = 10
    inputs = 1
elif dataset is 'CIFAR-100':    # train with CIFAR-100
    outputs = 100
    inputs = 3


if net is BBBLeNet:
    resize = 32
elif net is BBBAlexNet:
    resize = 227

'''
LOADING DATASET
'''

if net is BBBLeNet:
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    transforms.Lambda(lambda x: x + noise * torch.randn(x.size()))])
    train_dataset = dsets.MNIST(root="data", download=True,
                                transform=transform)
    val_dataset = dsets.MNIST(root="data", download=True, train=False,
                              transform=transform)
elif net is BBBAlexNet:
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    transforms.Lambda(lambda x: x + noise * torch.randn(x.size()))])
    train_dataset = dsets.CIFAR100(root="data", download=True, transform=transform)
    val_dataset = dsets.CIFAR100(root='data', download=True, train=False, transform=transform)

'''
MAKING DATASET ITERABLE
'''

print('length of training dataset:', len(train_dataset))
n_iterations = num_epochs * (len(train_dataset) / batch_size)
n_iterations = int(n_iterations)
print('Number of iterations: ', n_iterations)

loader_train = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

loader_val = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


# enable loading of weights to transfer learning
def cnnmodel(pretrained, task):
    model = net(outputs=outputs, inputs=inputs)

    if pretrained:
        if is_training:
            # load pretrained posterior distribution of one task as prior of next task
            with open("results/weights_{}.pkl".format(task-1), "rb") as previous:
                d = pickle.load(previous)
                model.load_prior(d)
        else:
            # evaluate accuracy of previous task
            with open("results/weights_{}.pkl".format(task), "rb") as previous:
                d = pickle.load(previous)
                model.load_prior(d)

    return model


'''
INSTANTIATE MODEL
'''

model = cnnmodel(pretrained=pretrained, task=task)

if cuda:
    model.cuda()

'''
INSTANTIATE VARIATIONAL INFERENCE AND OPTIMISER
'''
vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())
optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

'''
check parameter matrix shapes
'''

# how many parameter matrices do we have?
print('Number of parameter matrices: ', len(list(model.parameters())))

for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

'''
TRAIN MODEL
'''

if is_training:
    logfile = os.path.join('diagnostics_{}.txt'.format(task))
else:
    logfile = os.path.join('diagnostics_{}_eval.txt'.format(task))

with open(logfile, 'w') as lf:
    lf.write('')


def run_epoch(loader, epoch, is_training=False):
    m = math.ceil(len(loader.dataset) / loader.batch_size)

    accuracies = []
    likelihoods = []
    kls = []
    losses = []
    mean_fc = []
    var_fc = []
    mean_conv = []
    var_conv = []

    for i, (images, labels) in enumerate(loader):
        # Repeat samples (Casper's trick)
        if net is BBBAlexNet:
            x = images.view(-1, inputs, 227, 227).repeat(num_samples, 1, 1, 1)
            y = labels.repeat(num_samples)
        elif net is BBBLeNet:
            x = images.view(-1, inputs, 32, 32).repeat(num_samples, 1, 1, 1)
            y = labels.repeat(num_samples)

        if cuda:
            x = x.cuda()
            y = y.cuda()

        if beta_type is "Blundell":
            beta = 2 ** (m - (i + 1)) / (2 ** m - 1)
        elif beta_type is "Soenderby":
            beta = min(epoch / (num_epochs//4), 1)
        elif beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0

        logits, kl = model.probforward(x)
        loss = vi(logits, y, kl, beta)
        ll = -loss.data.mean() + beta*kl.data.mean()

        if is_training:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        _, predicted = logits.max(1)
        accuracy = (predicted.data.cpu() == y.cpu()).float().mean()
        mean_fc2 = model.fc2.qw_mean.data
        var_fc2 = model.fc2.qw_logvar.data
        mean_conv2 = model.conv2.qw_mean.data
        var_conv2 = model.conv2.qw_logvar.data

        accuracies.append(accuracy)
        losses.append(loss.data.mean())
        kls.append(beta*kl.data.mean())
        likelihoods.append(ll)
        mean_fc.append(mean_fc2)
        var_fc.append(var_fc2)
        mean_conv.append(mean_conv2)
        var_conv.append(var_conv2)

    diagnostics = {'loss': sum(losses)/len(losses),
                   'acc': sum(accuracies)/len(accuracies),
                   'kl': sum(kls)/len(kls),
                   'likelihood': sum(likelihoods)/len(likelihoods),
                   'mean_fc': mean_fc, 'var_fc': var_fc,
                   'mean_conv': mean_conv, 'var_conv': var_conv}

    return diagnostics


for epoch in range(num_epochs):
    if is_training is True:
        diagnostics_train = run_epoch(loader_train, epoch, is_training=True)
        diagnostics_val = run_epoch(loader_val, epoch)
        diagnostics_train = dict({"type": "train", "epoch": epoch}, **diagnostics_train)
        diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)
        print(diagnostics_train)
        print(diagnostics_val)

        with open(logfile, 'a') as lf:
            lf.write(str(diagnostics_train))
            lf.write(str(diagnostics_val))
    else:
        diagnostics_val = run_epoch(loader_val, epoch)
        diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)
        print(diagnostics_val)

        with open(logfile, 'a') as lf:
            lf.write(str(diagnostics_val))

'''
SAVE PARAMETERS
'''

if save_model:
    if is_training:
        with open("/results/weights_{}.pkl".format(task), "wb") as wf:
            pickle.dump(model.state_dict(), wf)

