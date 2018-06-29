import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet34
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np



def main():

    best_prec1 = 0.
    model = resnet34(pretrained = True)
    model.avgpool = nn.AvgPool2d(7)
    model.fc = nn.Linear(model.fc.in_features, 100)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    critearion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=0.004)

    epochs = 60
    traindir = './train'
    testdir = './val'

    traindata = datasets.ImageFolder(traindir,transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))


    traindataloader = data.DataLoader(traindata,batch_size=128,shuffle=True,num_workers=4,pin_memory=True, sampler=None)

    testdata = datasets.ImageFolder(testdir,transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
    testdataloader = data.DataLoader(testdata,batch_size=64,shuffle=True,num_workers=4)


    for epoch in range(epochs):

        adjust_learning_rate(optimizer,epoch)

        print('#' * 20)
        train(model,traindataloader,optimizer,critearion,device,epoch)
        prec1=validate(model,testdataloader,device,critearion,epoch)

        is_best = prec1 > best_prec1
        if is_best ==True:
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
            print ('save model')
        print('&' * 20)
        print ('\n')



def train(model,traindataloader,optimizer,critearion,device,epoch):
    start = time.time()
    model.train()

    print('on train data')

    for i ,(im,label) in enumerate(traindataloader):
        im = im.to(device)
        label = label.to(device)
        out = model(im)
        optimizer.zero_grad()
        loss = critearion(out,label)
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            timecost = time.time()-start
            print('Epoch {0}\t / dataset {1}\t  timecost{timecost: .2f}\t Loss{loss: .5f}\t'.format(epoch,len(traindataloader),timecost = timecost,loss = loss))



def validate(model,testdataloader,device,criterion,epoch):
    model.eval()
    start = time.time()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print('on test data')
    for i ,(im,label) in enumerate(testdataloader):
        im = im.to(device)
        label = label.to(device)
        out = model(im)
        loss = criterion(out,label)
        imdex = 140
        lenth = im.shape[0]
        # measure accuracy and record loss
        prec1, prec5 = accuracy(out.data, label, topk=(1, 5))
        losses.update(loss.item(), im.size(0))
        top1.update(prec1[0], im.size(0))
        top5.update(prec5[0], im.size(0))

        if top1.avg>=50.:
            fig = plt.figure(1)
            for k in range(4):
                x = random.randint(0, lenth-1)
                inp = im[x]
                inp = inp.to('cpu').numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = std * inp + mean
                showim = np.clip(inp, 0, 1)

                showoutvalue = out[x].data.to('cpu').numpy().tolist()
                showout = showoutvalue.index(max(showoutvalue))
                showimlabel = label.to('cpu').data[x].item()
                imdex += 1
                ax = fig.add_subplot(imdex)
                ax.imshow(showim)
                ax.set_title('pre:' + str(showout) + 'GT:' + str(showimlabel))
                plt.axis('off')
            plt.show()
            plt.clf()


        if i %3 ==0:
            timecost = time.time() - start
            print('Epoch {0}\t / dataset {1}\t  timecost{timecost: .2f}\t Loss{loss: .5f}\t top1{top1: .2f}\t top5{top5: .2f}'.format(epoch,len(testdataloader),
                timecost = timecost,loss = loss,top1 = top1.avg,top5 =top5.avg ))
    return top1.avg



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 20:
        lr=0.001
    if (epoch>=20 and epoch<40):
        lr=0.001 * 0.1
    if (epoch>=40 and epoch<60):
        lr=0.001 * 0.1 * 0.1
    if epoch>=60:
        lr=0.001 * 0.1 * 0.1 * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) #pre is the index of topk element
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ =='__main__':
    main()