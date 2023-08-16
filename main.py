import sys
sys.path.append("../CG-like-Adam")
sys.path.append("../fig")
import os
from check_gpu_available import gpu_available
from send_notice import send_notice
from CG_like_Adam import CG_like_Adam
from coba import coba as CoBA
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import pandas as pd
import datetime
from time import strftime
from multiprocessing import Pool, Manager



# VGG-19
class VGG19(nn.Module):
    # initialize model
    def __init__(self, img_size:int=224, input_channel:int=3, n_class:int=2):
        super().__init__()
        self.num_class = n_class
        self.input_channel = input_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.fc17 = nn.Sequential(
            nn.Linear(int(512 * img_size * img_size / 32 / 32), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)  # 默认就是0.5
        )

        self.fc18 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.fc19 = nn.Sequential(
            nn.Linear(4096, self.num_class)
        )

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
                          self.conv8, self.conv9, self.conv10, self.conv11, self.conv12, self.conv13, self.conv14,
                          self.conv15, self.conv16]

        self.fc_list = [self.fc17, self.fc18, self.fc19]

        print("VGG-19 Model Initialize Successfully!")

    # forward
    def forward(self, x):
        for conv in self.conv_list:    # 16 CONV
            x = conv(x)
        output = x.view(x.size()[0], -1)
        for fc in self.fc_list:        # 3 FC
            output = fc(output)
        return output



class ResBlk(nn.Module):
    # resnet block
    def __init__(self, ch_in: int, ch_out: int, stride: int=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] -> [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2),
                nn.BatchNorm2d(ch_out))

    def forward(self, x):
        # x:[b, ch, h, w]
        out =F.relu(self.bn1(self.conv1(x)))
        out =self.bn2(self.conv2(out))
        # short cut
        out = self.extra(x) + out
        # out = F.relu(out)
        return out



# ResNet34
class ResNet34(nn.Module):
    def __init__(self, n_class:int=10, input_channel:int=3):
        super(ResNet34, self).__init__()
        self.num_class = n_class
        self.input_channel = input_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        
        # followed 4 blocks
        # [b, 64, h, w] -> [b, 128, h, w]
        # self.blk1 = ResBlk(16, 16)
        self.blk1 = nn.Sequential(
            ResBlk(64, 64, 1),
            ResBlk(64, 64, 1),
            ResBlk(64, 64, 1))
        
        # [b, 128, h, w] -> [b, 256, h, w]
        # self.blk2 = ResBlk(16, 32)
        self.blk2 = nn.Sequential(
            ResBlk(64, 128, 2),
            ResBlk(128, 128, 1),
            ResBlk(128, 128, 1),
            ResBlk(128, 128, 1))
        
        # [b, 256, h, w] -> [b, 512, h, w]
        # self.blk3 = ResBlk(128, 256)
        self.blk3 = nn.Sequential(
            ResBlk(128, 256, 2),
            ResBlk(256, 256, 1),
            ResBlk(256, 256, 1),
            ResBlk(256, 256, 1),
            ResBlk(256, 256, 1),
            ResBlk(256, 256, 1))

        # [b, 512, h, w] -> [b, 1024, h, w]
        # self.blk4 = ResBlk(256, 512)
        self.blk4 = nn.Sequential(
            ResBlk(256, 512, 2),
            ResBlk(512, 512, 1),
            ResBlk(512, 512, 1))
        
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.outlayer = nn.Linear(512, self.num_class)

        print("ResNet-34 Model Initialize Successfully!")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # [b, 64, h, w] -> [b, 128, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x



def train(model, device, train_data, optimizer, n_class): 
    criterion = nn.CrossEntropyLoss().to(device)

    model.train()
    total_correct = 0
    total_num = 0
    max_batch_idx = len(train_data)
    for batch_idx, (x, label) in enumerate(train_data):
        x, label = x.to(device), label.to(device)
        logits = model(x)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = logits.argmax(dim=1)
        total_correct += torch.eq(pred, label).float().sum().item()
        total_num += x.size(0)
        acc = total_correct / total_num
        if batch_idx == max_batch_idx-1:
        # if (batch_idx % 500 == 0 or batch_idx == max_batch_idx-1):
            print('[GPU {}] Train set [batch index {}]: Loss: {:.6f}, Accuracy: {:.2f}%'.format(device, batch_idx, loss.data.item(), acc*100.))
    return loss.data.item(), acc



def test(model, device, test_data, n_class):
    data_size = len(test_data.dataset)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)  # 累加loss

    test_loss = 0.0 
    total_correct = 0
    for batch_idx, (x, y) in enumerate(test_data):
        x, label = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
        test_loss += criterion(logits, label)
        pred = logits.argmax(dim=1)
        # [b] vs [b] -> scalar tensor
        total_correct += torch.eq(pred, label).float().sum().item()
    acc = total_correct / data_size
    test_loss /= data_size
    if device == 0:
        print('\nTest set: Average loss: {:.6f}, Accuracy: {:.2f}%'.format(test_loss, 100. * acc))
    return acc



def load_data(savepath: str, datatype: str='CIFAR10', batch_size: int=32, train: bool=True, download: bool=True):
    check_path(savepath)
    if datatype == 'CIFAR10' or datatype == 'cifar10':
        data = datasets.CIFAR10(savepath, train, transform = transforms.Compose([
            transforms.Resize([32, 32]),transforms.ToTensor()]), download = download)
    elif datatype == 'CIFAR100' or datatype == 'cifar100':
        data = datasets.CIFAR100(savepath, train, transform = transforms.Compose([
            transforms.Resize([32, 32]),transforms.ToTensor()]), download = download)
    elif datatype == 'MNIST' or datatype == 'mnist':
        data = datasets.MNIST(savepath, train, transform = transforms.Compose([
            transforms.Resize([32, 32]),transforms.ToTensor()]), download = download)
    else:
        raise Exception("Unknow downloaded data type: "+str(datatype))
    return DataLoader(data, batch_size=batch_size, shuffle=True)



def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)



def main(traindata, testdata, lock, n_class:int, input_channel:int, RUN_1:bool, lr:float=1e-3,  
        amsgrad:bool=True, gammatype:str='PRP', optimizer_type:str='CG_like_Adam', Epoch:int=200):
    
    lock.acquire()

    if RUN_1:
        gpuid = gpu_available(gpu_usage_demand=80, men_usage_demand=78, men_demand=24382, interval=1, reverse=False)
    else:
        gpuid = gpu_available(gpu_usage_demand=80, men_usage_demand=78, men_demand=24382, interval=120, reverse=False)
    device_id = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")
    
    # create model and move it to GPU with id rank or to CPU
    # Decide which model to use here
    # model = ResNet34(n_class=n_class, input_channel=input_channel).to(device_id)
    model = VGG19(img_size=32, n_class=n_class, input_channel=input_channel).to(device_id)

    start = time.perf_counter()

    now=datetime.datetime.now().strftime("%Y-%m-%d")

    csvresultspath = './results-'+datatype+'/'+now+'/csv/'
    figresultspath = './results-'+datatype+'/'+now+'/fig/'

    if optimizer_type == 'Adam':
        print('Using Adam optimizer.')
        optimizer = optim.Adam(model.parameters(), amsgrad=amsgrad)
        csvpath = csvresultspath+'Adam_epoch'+str(Epoch)+'_lr'+str(lr)+'.csv'
        figpath = figresultspath+'Adam_epoch'+str(Epoch)+'_lr'+str(lr)+'.png'

    elif optimizer_type == 'CoBA':
        print('Using CoBA optimizer.')
        optimizer = CoBA(model.parameters(), lr=lr, gammatype=gammatype, amsgrad=amsgrad)
        csvpath = csvresultspath+'CoBA_epoch'+str(Epoch)+'_lr'+str(lr)+'_'+gammatype+'.csv'
        figpath = figresultspath+'CoBA_epoch'+str(Epoch)+'_lr'+str(lr)+'_'+gammatype+'.png'

    elif optimizer_type == 'CG_like_Adam':
        print('Using CG_like_Adam optimizer.')
        optimizer = CG_like_Adam(model.parameters(), lr=lr, amsgrad=amsgrad, gammatype=gammatype)
        path_ = 'CG_like_Adam-'
        csvpath = csvresultspath + path_ +'epoch'+str(Epoch)+'_lr'+str(lr)+'_'+gammatype+'.csv'
        figpath = figresultspath + path_ +'epoch'+str(Epoch)+'_lr'+str(lr)+'_'+gammatype+'.png'
    else:
        raise Exception("Unknow optimizer type: "+str(optimizer_type))

    lock.release()

    best_acc = 0.0
    train_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(1, Epoch+1):
        print("Current epoch is {}.".format(epoch))
        # adjust_learning_rate(optimizer, current_epoch=epoch, step_size=60, lr_decay=1) # 每60个epoch，学习率降低0.1倍
        trainloss, trainacc = train(model, device_id, traindata, optimizer, n_class)
        testacc = test(model, device_id, testdata, n_class)
        train_loss.append(trainloss)
        train_acc.append(trainacc)
        test_acc.append(testacc)
        if best_acc < testacc:
            best_acc = testacc
        print("[Epoch {}]Test accuracy is {:.2f}%, current best test accuracy is {:.2f}%.\n".format(epoch, testacc*100., best_acc*100.)) 

    end = time.perf_counter()
    print("[GPU {}]time cost of running {}".format(device_id, round(end-start,4)), 'seconds')
    
    check_path(csvresultspath)
    check_path(figresultspath)

    # save classification result to csv
    df = pd.DataFrame(data={'train_loss':train_loss, 'train_acc':train_acc, 'test_acc':test_acc})
    df.to_csv(csvpath, index=False)
    
    # plot figure
    plot_figure(Epoch, train_loss, train_acc, test_acc, figpath)



def plot_figure(epoch, train_loss, train_acc, test_acc, figpath):
    plt.figure(facecolor="lightgray")
    
    x_ticks = range(0,epoch+1,int(epoch/10))
    plt.xticks(x_ticks)
    aix = plt.gca()
    aix.xaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.grid(linestyle=":")
    
    plt.plot(train_loss, label="train_loss")
    plt.plot(train_acc, label="train_acc")
    plt.plot(test_acc, label="test_acc")
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    # save figure
    plt.savefig(figpath, dpi=300)
    plt.show()



def err_call_back(err):
    print(f'Error：{str(err)}')



def adjust_learning_rate(optimizer, current_epoch:int, step_size:int=150, lr_decay:float=0.1, reset:bool=False):
    for param_group in optimizer.param_groups:
        if current_epoch>0 and (current_epoch % step_size)==0:
            param_group['lr'] *= lr_decay
            print(param_group['lr'])

    if reset and current_epoch>0 and (current_epoch % step_size)==0:
        optimizer.reset()





if __name__ == '__main__':
    start = time.perf_counter()
    # load dataset
    batchsz = 512

    datatype = 'cifar10'

    if datatype == 'cifar10':
        datapath = '../data/CIFAR/cifar10'
        n_class = 10
        input_channel = 3
    elif datatype == 'cifar100':
        datapath = '../data/CIFAR/cifar100'
        n_class = 100
        input_channel = 3
    elif datatype == 'MNIST':
        datapath = '../data/MNIST/'
        n_class = 10
        input_channel = 1

    traindata = load_data(savepath=datapath, datatype=datatype, batch_size=batchsz, train=True)
    testdata = load_data(savepath=datapath, datatype=datatype, batch_size=batchsz, train=False)
    
    process = 6
    pool = Pool(processes = process)
    i = 0
    epochs = 200
    lock = Manager().Lock()
    lrs = (1e-3, 1e-4, 1e-5, 1e-6,)
    run_1 = False
    if run_1:
        lr = 1e-6
        pool.apply_async(main, args=(traindata, testdata, lock, n_class, input_channel, 1e-20, run_1, lr, 
                                    False, 'FR', 'Adam', False, epochs), error_callback=err_call_back)
        i += 1
        print('====== Process[{}] apply_async {}  ======'.format(os.getpid(),i))
        print('Run one program.')
    else:
        for lr in lrs:
            for optimizer_type in ('Adam', 'CoBA', 'CG_like_Adam',):
                if optimizer_type == 'Adam':
                    pool.apply_async(main, args=(traindata, testdata, lock, n_class, input_channel, run_1, lr, 
                                                False, None, optimizer_type, epochs), error_callback=err_call_back)
                    i += 1
                    print('====== Process[{}] apply_async {}  ======'.format(os.getpid(),i))
                elif optimizer_type == 'CoBA':
                    for gammatype in ('FR', 'PRP', 'HS', 'DY', 'HZ',):
                        pool.apply_async(main, args=(traindata, testdata, lock, n_class, input_channel, run_1, lr, 
                                                    False, gammatype, optimizer_type, epochs), error_callback=err_call_back)
                        i += 1
                        print('====== Process[{}] apply_async {}  ======'.format(os.getpid(),i))
                elif optimizer_type == 'CG_like_Adam':
                    for gammatype in ('FR', 'PRP', 'HS', 'DY', 'HZ', 'MFR', 'MPRP',):
                        if gammatype in ('MFR', 'MPRP',):
                            continue
                        pool.apply_async(main, args=(traindata, testdata, lock, n_class, input_channel, run_1, lr, 
                                                     False, gammatype, optimizer_type, epochs), error_callback=err_call_back)
                        i += 1
                        print('====== Process[{}] apply_async {}  ======'.format(os.getpid(),i))
    pool.close()
    pool.join()
    end = time.perf_counter()
    run_time = round(end-start,4)
    m, s = divmod(run_time, 60)
    h, m = divmod(m, 60)
    content = "total time cost of main program: %02d:%02d:%02d" % (h, m, s)
    print(content)
    # please replace the token of pushplus by yours
    # you can get your own token on this website http://www.pushplus.plus/
    try:
        send_notice(token="", content=content)
    except:
        print("please replace the token of pushplus by yours.")
        print("you can get your own token on this website http://www.pushplus.plus/")
    print('Mainprocess is end.')
    
