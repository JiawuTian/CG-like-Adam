import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


colors = {'1e-2':'blue', '1e-3':'green', '1e-4':'purple', '1e-5':'orange', '1e-6':'brown'}
baselines = ('CoBA', 'Adam',)
# lrs = ('1e-2', '1e-3', '1e-4', '1e-5', '1e-6',)
lrs = ('1e-3', '1e-4', '1e-5', '1e-6',)
len_lrs = len(lrs)
len_colors = len(colors)
if len_colors < len_lrs:
    raise ValueError("colors dict does not match the lrs.")
gammatypes = ('FR', 'PRP', 'HS', 'DY', 'HZ', 'MFR', 'MPRP',)
date = '2023-05-19'
datatype = 'cifar100'
epoch = 200
len_lrs = len(lrs)


figpath = './' + datatype + '/' + date + '-lr/'
if not os.path.exists(figpath):
    os.makedirs(figpath)


for baseline in baselines:
    for gammatype in gammatypes:
        for datatype1 in ('train_loss', 'train_log_loss', 'train_acc', 'test_acc',):
            name = datatype1.split('_')
            ylabel = name[0]+ ' ' + name[1]
            if datatype1 == 'train_log_loss':
                ylabel += ' loss'
            titlename = 'Epoch '+ ylabel + '  ' + gammatype
            fig_save = baseline + '_CG-like-Adam_' + gammatype + '_' + datatype1 + '.pdf'
            plt.figure(titlename)
            plt.title(titlename, fontsize=20)
            plt.grid(linestyle=":")
            close1 = 0
            close2 = 0
            close3 = 0
            for lr in lrs:
                CG_like_Adam_file = '../ImageClassification/results-'+datatype+'/'+date+'/csv/CG_like_Adam-epoch'+str(epoch)+'_lr'+str(float(lr))+'_'+gammatype+'.csv'
                CG_like_Adam_legend = f'CG-like Adam lr={lr}'
                try:
                    CG_like_Adam_df = pd.read_csv(CG_like_Adam_file)
                    if datatype1 == 'train_log_loss':
                        plt.plot(np.log(CG_like_Adam_df['train_loss']), linewidth=1.1, color=colors[lr], linestyle='-', label=CG_like_Adam_legend)
                    else:
                        plt.plot(CG_like_Adam_df[datatype1], linewidth=1.1, color=colors[lr], linestyle='-', label=CG_like_Adam_legend)
                except FileNotFoundError:
                    close1 += 1
                    pass
                if baseline == 'Adam':
                    Adam_file = '../ImageClassification/results-'+datatype+'/'+date+'/csv/Adam_epoch'+str(epoch)+'_lr'+str(float(lr))+'.csv'
                    Adam_legend = f'Adam lr={lr}'
                    try:
                        Adam_df = pd.read_csv(Adam_file)
                        if datatype1 == 'train_log_loss':
                            plt.plot(np.log(Adam_df['train_loss']), linewidth=0.8, color=colors[lr], linestyle='--', label=Adam_legend)
                        else:
                            plt.plot(Adam_df[datatype1], linewidth=0.8, color=colors[lr], linestyle='--', label=Adam_legend)
                    except FileNotFoundError:
                        close2 += 1
                        pass
                elif baseline == 'CoBA':
                    CoBA_file = '../ImageClassification/results-'+datatype+'/'+date+'/csv/CoBA_epoch'+str(epoch)+'_lr'+str(float(lr))+'_'+gammatype+'.csv'
                    CoBA_legend = f'CoBA lr={lr}'
                    try:
                        CoBA_df = pd.read_csv(CoBA_file)
                        if datatype1 == 'train_log_loss':
                            plt.plot(np.log(CoBA_df['train_loss']), linewidth=0.8, color=colors[lr], linestyle='--', label=CoBA_legend)
                        else:
                            plt.plot(CoBA_df[datatype1], linewidth=0.8, color=colors[lr], linestyle='--', label=CoBA_legend)
                    except FileNotFoundError:
                        close3 += 1
                        pass
                    
            if close1 == len_lrs or close2 == len_lrs or close3 == len_lrs:
                plt.close()
            else:
                plt.xlabel('epoch', fontsize=12)
                plt.ylabel(ylabel, fontsize=12)
                plt.legend()
                plt.tight_layout()
                plt.savefig(figpath+fig_save, dpi=300, format="pdf")
                plt.show()
                plt.close('all')

