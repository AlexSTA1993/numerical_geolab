'''
Created on May 11, 2022

@author: Alexandros Stathas

Import matplotlib parameters and custom functions for the creation of beautiful plots
'''
import os
# %matplotlib inline
import matplotlib as mpl
import numpy as np


import matplotlib.pyplot as plt
from itertools import cycle, tee

mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100

# mpl.rcParams['font.size'] = 20
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

colors = cycle(['r','g','b','m','k'])
markers = cycle(['o','s','v', 'd','*'])
linestyles = cycle(['-', '--', '-.', ':',(0, (1, 1))])

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


font = {'family' : 'serif',
        'size'   : 25}

mpl.rc('font', **font)

def empty_plot():
    plt.plot()
    plt.show()


def object_plot_axes(x_txt='x_text', y1_txt='y_text',color1='', y2_txt='y_text',color2='', title='title',mode='1'):
    if color1=='':
        color1=next(colors)

    fig1, ax1 = plt.subplots()
    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax1.grid(True)
    ax1.set_xlabel(x_txt)
    ax1.set_ylabel(y1_txt,color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title(title)

    if mode=='1':
        return fig1, ax1, color1

    elif mode=='2':
        if color2=='':
            color2=next(colors)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
        
        ax2.set_ylabel(y2_txt, color=color2)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color2)
        fig1.tight_layout()    
        return fig1, ax1, ax2, color1, color2
    else:
        return

def object_plot( obj_x, obj_F1,obj_F2,*args,plot_mode='normal',mode='1',color1='',color2='',label_string=''):
    if plot_mode=='normal':
        if color1=='':
            color1=next(colors)
        
        args[0].plot(obj_x, obj_F1,label = label_string,c=color1, linestyle = next(linestyles), marker=next(markers), markersize=10,markerfacecolor="None",
         markeredgecolor=color1, markeredgewidth=2, markevery=1)

        if mode=='1':
            return
        elif  mode=='2':
 
            if color2=='':
                color2=next(colors)
            args[1].plot(obj_x, obj_F2,label = label_string,c=color2, linestyle = next(linestyles), marker=next(markers), markersize=10, markevery=1)

        elif  mode=='3':
            if color2=='':
                color2=next(colors)
            if isinstance(args[1],str)==False:
                args[1].set_ylim(min(obj_F2),1.1*max(obj_F2))
                args[1].set_yticks(np.linspace(min(obj_F2),max(obj_F2), len(args[0].get_yticks())))
            args[0].plot(args[2], np.array(obj_F2)-min(obj_F2),label = label_string,c=color2, linestyle = next(linestyles), marker=next(markers), markersize=10, markevery=1)
    
    elif plot_mode=='scatter':
        args[0].plot(obj_x, obj_F1,label = label_string,c=color1[0], linestyle = '-')

        args[1].scatter(args[2],obj_F2,label = label_string, edgecolors=color2[0], marker='o',facecolors='none')#, markevery=0.5)
    

        
    else:
        if color2=='':
            color2=next(colors)
        args[0].plot(args[2], obj_F2,label = label_string,c=color2, linestyle = next(linestyles), marker=next(markers), markersize=10, markevery=1)
    return

def object_plot_doule(ax1,x1,y1,y2,ax2,x2,y3,y4, mode='1',color1=['r','b'],color2=['m','c'],label_string=''):

    if mode=='1':
        ax1.plot(x1,y1,label = label_string,c=color1[0], linestyle = next(linestyles), marker=next(markers), markersize=10, markevery=1)
        ax1.plot(x1,y2,label = label_string,c=color1[1], linestyle = next(linestyles), marker=next(markers), markersize=10, markevery=1)

        ax2.plot(x1,y3,label = label_string,c=color2[0], linestyle = next(linestyles), marker=next(markers), markersize=10, markevery=1)
        ax2.plot(x1,y4,label = label_string,c=color2[1], linestyle = next(linestyles), marker=next(markers), markersize=10, markevery=1)

    elif mode=='2':
        ax1.plot(x1,y1,label = label_string,c=color1[0], linestyle = '-')
        ax1.plot(x1,y2,label = label_string,c=color1[1], linestyle = '-')

        ax2.scatter(x2,y3,label = label_string, edgecolors=color2[0], marker='o',facecolors='none')#, markevery=0.5)
        ax2.scatter(x2,y4,label = label_string, edgecolors=color2[1], marker='o',facecolors='none')#, markevery=0.5)
        
    else:
        print('set_mode_to: 1 or 2')
        return

    return

def show_plot():
    plt.show()

def plot_legends(path_out, fig,filename = ' ', mode='1'):  
    '''
    This function passes the legend titles for the plots and saves the figures
    '''
    if mode=='1':
        fname =  os.path.join(path_out,filename+".svg")
        fname1 = os.path.join(path_out,filename+".pdf")

    fig.savefig(fname, dpi=150, bbox_inches='tight')
    fig.savefig(fname1, dpi=150, bbox_inches='tight')

    return   