import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# colors = {'Adam':'black','FR':'coral', 'PRP':'limegreen', 'HS':'deepskyblue', 'DY':'violet', 'HZ':'teal'}
colors = {'Adam':'black','FR':'deepskyblue', 'PRP':'green', 'HS':'purple', 'DY':'orange', 'HZ':'brown'}
lr = '1e-3'
gammatypes = ('FR', 'PRP', 'HS', 'DY', 'HZ',)
len_gammas = len(gammatypes)
len_colors = len(colors)
if len_colors < len_gammas:
    raise ValueError("colors dict does not match the gammatypes.")

date = '2023-05-19'
datatype = 'cifar100'
epoch = 200
start_epoch = 25

figpath = './' + datatype + '/' + date + '-gamma/'
if not os.path.exists(figpath):
    os.makedirs(figpath)


for datatype1 in ('train_loss', 'train_log_loss', 'train_acc', 'test_acc',):
    name = datatype1.split('_')
    ylabel = name[0]+ ' ' + name[1]
    if datatype1 == 'train_log_loss':
        ylabel += ' loss'
    titlename = 'Epoch '+ ylabel
    fig_save = 'Adam_CG-like-Adam_' + datatype1 + '.pdf'
    plt.figure(titlename)
    plt.title(titlename, fontsize=20)
    plt.grid(linestyle=":")
    close1 = 0
    close2 = 0

    Adam_file = '../ImageClassification/results-'+datatype+'/'+date+'/csv/Adam_epoch'+str(epoch)+'_lr'+str(float(lr))+'.csv'
    try:
        Adam_df = pd.read_csv(Adam_file)
        if datatype1 == 'train_log_loss':
            plt.plot(np.log(Adam_df['train_loss']), linewidth=1.5, color=colors['Adam'], linestyle='--', label='Adam')
        elif datatype1 == 'train_loss':
            plt.plot(Adam_df['train_loss'], linewidth=1.5, color=colors['Adam'], linestyle='--', label='Adam')
        else:
            plt.plot(Adam_df[datatype1][start_epoch:], linewidth=1.5, color=colors['Adam'], linestyle='--', label='Adam')
    except FileNotFoundError:
        close1 += 1
        pass

    for gammatype in gammatypes:
        CG_Adam_file = '../ImageClassification/results-'+datatype+'/'+date+'/csv/CG_like_Adam-epoch'+str(epoch)+'_lr'+str(float(lr))+'_'+gammatype+'.csv'
        try:
            CG_Adam_df = pd.read_csv(CG_Adam_file)
            if datatype1 == 'train_log_loss':
                plt.plot(np.log(CG_Adam_df['train_loss']), linewidth=0.9, color=colors[gammatype], linestyle='-', label=gammatype)
            elif datatype1 == 'train_loss':
                plt.plot(CG_Adam_df['train_loss'], linewidth=0.9, color=colors[gammatype], linestyle='-', label=gammatype)
            else:
                plt.plot(CG_Adam_df[datatype1][start_epoch:], linewidth=0.9, color=colors[gammatype], linestyle='-', label=gammatype)
        except FileNotFoundError:
            close2 += 1
            pass

            
    if close1 == 1 or close2 == len_gammas:
        plt.close()
    else:
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figpath+fig_save, dpi=300, format="pdf")
        plt.show()
        plt.close('all')

