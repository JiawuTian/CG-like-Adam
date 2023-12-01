import sys
sys.path.append("../CG-like-Adam")
sys.path.append("../fig")
import os
from check_gpu_available import gpu_available
from send_notice import send_notice
from CG_like_Adam import CG_like_Adam
from coba import coba
from model import VGG19, ResNet34
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import pandas as pd
import datetime
from multiprocessing import Pool, Manager



def train(model, device, train_data, optimizer): 
    criterion = nn.CrossEntropyLoss().to(device)
    model.train()
    total_correct = 0
    total_num = 0
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
    return loss.data.item(), acc



def test(model, device, test_data):
    data_size = len(test_data.dataset)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    test_loss = 0.0 
    total_correct = 0
    for batch_idx, (x, y) in enumerate(test_data):
        x, label = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
        test_loss += criterion(logits, label)
        pred = logits.argmax(dim=1)
        total_correct += torch.eq(pred, label).float().sum().item()
    acc = total_correct / data_size
    test_loss /= data_size
    return test_loss.data.item(), acc



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



def err_call_back(err):
    print(f'Error：{str(err)}')



def main(traindata, testdata, lock, model_type:str, n_class:int, input_channel:int, RUN_1:bool, 
         lr:float=1e-3, amsgrad:bool=True, gammatype:str='PRP', optimizer_type:str='CG_like_Adam', 
         Epoch:int=200):
    
    lock.acquire()

    if RUN_1:
        gpuid = gpu_available(gpu_usage_demand=80, men_usage_demand=78, men_demand=24382, interval=1, reverse=False)
    else:
        gpuid = gpu_available(gpu_usage_demand=80, men_usage_demand=78, men_demand=24382, interval=120, reverse=False)
    device_id = torch.device("cuda:"+str(gpuid) if torch.cuda.is_available() else "cpu")
    
    # create model and move it to GPU with id rank or to CPU
    # Decide which model to use here
    if model_type == 'ResNet-34':
        model = ResNet34(n_class=n_class, input_channel=input_channel).to(device_id)
    elif model_type == 'VGG-19':
        model = VGG19(img_size=32, n_class=n_class, input_channel=input_channel).to(device_id)
    else:
        raise ValueError(f"Could NOT know model type {model_type}!")

    start = time.perf_counter()

    now=datetime.datetime.now().strftime("%Y-%m-%d")

    csvresultspath = './results-'+datatype+'/'+now+'/csv/'
    figresultspath = './results-'+datatype+'/'+now+'/fig/'

    if optimizer_type == 'Adam':
        print('Using Adam optimizer.\n')
        optimizer = optim.Adam(model.parameters(), amsgrad=amsgrad)
        csvpath = csvresultspath+'Adam_epoch'+str(Epoch)+'_lr'+str(lr)+'.csv'
        figpath = figresultspath+'Adam_epoch'+str(Epoch)+'_lr'+str(lr)+'.pdf'

    elif optimizer_type == 'coba':
        print('Using coba optimizer.\n')
        optimizer = coba(model.parameters(), lr=lr, gammatype=gammatype, amsgrad=amsgrad)
        csvpath = csvresultspath+'CoBA_epoch'+str(Epoch)+'_lr'+str(lr)+'_'+gammatype+'.csv'
        figpath = figresultspath+'CoBA_epoch'+str(Epoch)+'_lr'+str(lr)+'_'+gammatype+'.pdf'

    elif optimizer_type == 'CG_like_Adam':
        print('Using CG_like_Adam optimizer.\n')
        optimizer = CG_like_Adam(model.parameters(), lr=lr, amsgrad=amsgrad, gammatype=gammatype)
        path_ = 'CG_like_Adam-'
        csvpath = csvresultspath + path_ +'epoch'+str(Epoch)+'_lr'+str(lr)+'_'+gammatype+'.csv'
        figpath = figresultspath + path_ +'epoch'+str(Epoch)+'_lr'+str(lr)+'_'+gammatype+'.pdf'
    else:
        raise Exception("Unknow optimizer type: "+str(optimizer_type))

    lock.release()

    best_acc = 0.0
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(1, Epoch+1):
        if epoch % 10 == 0 or epoch == 1 or epoch == Epoch+1:
            print("Current epoch is {}.".format(epoch))
        trainloss, trainacc = train(model, device_id, traindata, optimizer)
        testloss, testacc = test(model, device_id, testdata)
        train_loss.append(trainloss)
        train_acc.append(trainacc)
        test_loss.append(testloss)
        test_acc.append(testacc)
        if best_acc < testacc:
            best_acc = testacc
        if epoch == 1 or epoch % 10 == 0 or epoch == Epoch+1:
            print('[Epoch {}]Train loss is {:.6f}, train accuracy is {:.2f}%'.format(epoch, trainloss, trainacc*100.))
            print('[Epoch {}]Test loss is {:.6f}, test accuracy is {:.2f}%'.format(epoch, testloss, testacc*100.))
            print("[Epoch {}]Current best test accuracy is {:.2f}%.\n".format(epoch, best_acc*100.))

    end = time.perf_counter()
    print("[GPU {}]time cost of running {}".format(device_id, round(end-start,4)), 'seconds')
    
    check_path(csvresultspath)
    check_path(figresultspath)

    # save classification result to csv
    df = pd.DataFrame(data={'train_loss':train_loss, 'train_acc':train_acc, 'test_loss':test_loss, 'test_acc':test_acc})
    df.to_csv(csvpath, index=False)
    
    # plot figure
    plot_figure(Epoch, train_loss, train_acc, test_loss, test_acc, figpath)



def plot_figure(epoch, train_loss, train_acc, test_loss, test_acc, figpath):
    plt.figure(facecolor="lightgray")
    
    x_ticks = range(1,epoch+1,int(epoch/10))
    plt.xticks(x_ticks)
    aix = plt.gca()
    aix.xaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.grid(linestyle=":")
    
    plt.plot(train_loss, label="train_loss")
    plt.plot(train_acc, label="train_acc")
    plt.plot(test_loss, label="test_loss")
    plt.plot(test_acc, label="test_acc")
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    # save figure
    plt.savefig(figpath, dpi=300)
    plt.show()




if __name__ == '__main__':
    start = time.perf_counter()

    # load dataset
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

    batchsz = 512
    traindata = load_data(savepath=datapath, datatype=datatype, batch_size=batchsz, train=True)
    testdata = load_data(savepath=datapath, datatype=datatype, batch_size=batchsz, train=False)
    
    process = 6
    pool = Pool(processes = process)
    i = 0
    epochs = 20
    lock = Manager().Lock()
    lrs = (1e-3, 1e-4, 1e-5, 1e-6,)
    run_1 = True
    model_type = 'ResNet-34'
    # model_type = 'VGG-19'
    if run_1:
        lr = 1e-3
        pool.apply_async(main, args=(traindata, testdata, lock, model_type, n_class, input_channel, run_1, lr, 
                                    True, 'FR', 'CG_like_Adam', epochs), error_callback=err_call_back)
        i += 1
        print('====== Process[{}] apply_async {}  ======'.format(os.getpid(),i))
        print('Run one program.')
    else:
        for lr in lrs:
            for optimizer_type in ('Adam', 'coba', 'CG_like_Adam',):
                if optimizer_type == 'Adam':
                    pool.apply_async(main, args=(traindata, testdata, lock, model_type, n_class, input_channel, run_1, lr, 
                                                True, None, optimizer_type, epochs), error_callback=err_call_back)
                    i += 1
                    print('====== Process[{}] apply_async {}  ======'.format(os.getpid(),i))
                elif optimizer_type == 'coba':
                    for gammatype in ('FR', 'PRP', 'HS', 'DY', 'HZ',):
                        pool.apply_async(main, args=(traindata, testdata, lock, n_class, input_channel, run_1, lr, 
                                                    True, gammatype, optimizer_type, epochs), error_callback=err_call_back)
                        i += 1
                        print('====== Process[{}] apply_async {}  ======'.format(os.getpid(),i))
                elif optimizer_type == 'CG_like_Adam':
                    for gammatype in ('FR', 'PRP', 'HS', 'DY', 'HZ', 'MFR', 'MPRP',):
                        if gammatype in ('MFR', 'MPRP',):
                            continue
                        pool.apply_async(main, args=(traindata, testdata, lock, model_type, n_class, input_channel, run_1, lr, 
                                                     True, gammatype, optimizer_type, epochs), error_callback=err_call_back)
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
    ## please replace the token of pushplus by yours
    ## you can get your own token on this website http://www.pushplus.plus/
    send_notice(token="", content=content)
    print('Mainprocess is end.')
    
