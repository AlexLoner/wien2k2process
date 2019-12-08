# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class Graph():
    
    def __init__(self, route):
        self.route = route
        
    def __lshift__(self, other):
        '''Merging two objects'''
        postfix = other.route.split('/')[-1].split('.')[0]
        other.data = other.data.rename(columns={i: i+'_'+postfix for i in other.data.columns})
        self.data = pd.concat((self.data, other.data), axis=1)
        
    
    def plot(self, col, names, xlab='', ylab='', xscale=None, yscale=None, figsize=(20,10), grid=True, colors=['green', 'violet'], 
             linewidth=4.0, alpha=0.5, fontsize=24, ylabel='left', xtickslabels=[0], ytickslabels=[0], zeroline=0, 
             location='best', legend_font=0,  x_axis=[], handlelength=None, save=None):
        '''
        Function plots several dos on one figure
        ----------
        data : list of DataFrames 
        col : list of columns that should be plot in respective DataFrame
        names : list of names for each plot
        xlab, ylab : signature of axis :: should be str
        yscale (xscale) : scale on the ordinat (absciss) axis :: tuple
        colors : list with colors for each curves
        alpha : coefficient intensity for filling under curves
        '''
#         plt.style.use('dark_background')
        plt.figure(figsize=figsize)
        if grid:
            plt.grid('grey', linestyle='--', linewidth=0.5)
        if ylabel == 'right':
            plt.tick_params(axis='y', labelleft='off', labelright='on', labelsize=fontsize-2)
        else:
            plt.tick_params(axis='y', labelleft='on', labelright='off', labelsize=fontsize-2)
            plt.ylabel(ylab, fontsize=fontsize)
        if len(xlab) > 0:
            plt.xlabel(xlab, fontsize=fontsize)
            plt.tick_params(axis='x', labelsize=fontsize-2)
        else:
            plt.xticks(fontsize=0)
        l = len(col)
                  
        cns = self.data.columns
        
        if  len(x_axis) == 0:
             x_axis = [cns[0]] * l
        
        
        for i in range(l):
            plt.plot(self.data[x_axis[i]], self.data[col[i]], colors[i], linewidth=linewidth)
            plt.fill_between(self.data[x_axis[i]], self.data[col[i]], color=colors[i], alpha=alpha)
#             if xscale:
#                 plt.xlim(xscale)
#                 # Вычисление верхней границы оси ординат
#                 cor = 1.1 * self.data[col[i]][self.data[cns[0]] < xscale[1]][self.data[cns[0]] > xscale[0]].max()
#                 if cor > ylim:
#                     ylim = cor
#                 plt.ylim(0, ylim)
            if legend_font == -1: legend_font = fontsize
            plt.legend(names, loc=location, fontsize=legend_font, handlelength=handlelength)
        
        if len(ytickslabels) == 0:
            plt.yticks(fontsize=0)
        else:
            plt.yticks(ytickslabels)
        
        if len(xtickslabels) == 0:
            plt.xticks(fontsize=0)
        
        if zeroline:
            plt.plot(self.data[cns[0]], np.zeros(len(self.data[cns[0]])), 'black', lw =0.5)
            plt.xlim(min(self.data[cns[0]]), max(self.data[cns[0]]))
        
        if xscale:
            plt.xlim(xscale)
        
        if yscale:
            plt.ylim(yscale)
        
        if save:
            plt.savefig(save, dpi=512)
            plt.close()

##################################################
##################################################
class Xas(Graph):
    def __init__(self, route):
        self.route = route
        self.data = self._convert()
        
    def get_columns(self):
        return self.data.columns
    
    def _convert(self):
        _file = [i.strip() for i in open(self.route)]
        s, gamma = [float(i.split(' ')[-1]) for i in _file[:2]]
        data = [map(float, i.split()) for i in _file[2:]]
        df = pd.DataFrame(data, columns=['Energy', 'Intensity'] + ['y' + str(i) for i in range(2, len(data[0]))])
        return df
    
##################################################
##################################################
class Optic(Graph):
    
    def __init__(self, route):
        '''
        route :: route to file
        '''
        self.route = route
        self.data = self._convert()
    
    def get_columns(self):
        '''
        Return columns names of your DataFrame
        '''
        return self.data.columns
    
    def _convert(self):
        f = open(self.route, 'r')
        _title = [next(f).split() for _ in range(3)]
        title = [_title[1][1]] + _title[1][3:]
        data = [map(float, line.split()) for line in f]
        df = pd.DataFrame(data=data, columns=title)
        return df

##################################################
##################################################
class Dos(Graph):
    def __init__(self, route, name, num):
        '''
        route :: route to file
        name :: name of *dos?ev file
        num :: number of *.dos?ev files
        '''
        self.route = route
        self.num = num 
        self.name = name
        self.data = self._merge()
    
    def get_columns(self):
        '''
        Return columns names of your DataFrame
        '''
        return list(self.data.columns)
        
    def convert_dos(self, route):
        '''
        Mapping dos-file в DataFrame
        '''
        f = open(route)
        _title = [next(f).strip() for _ in range(3)] 
        title = _title[2][1:].replace('-', '_').replace(':', '_').split()
        title = [title[i] if not title[i][0].isdigit() else title[i][2:] + title[i][0:1] for i in range(len(title))]
        data = [map(float, line.split()) for line in f]
        df = pd.DataFrame(data=data, columns=title)
        return df

    def _merge(self):
        '''
        Collect all DataFrame into big one
        '''
        routes_dos = [self.route + '/' + self.name + '.dos' + str(i) + 'ev' for i in range(1, self.num + 1)]
        dfs = [self.convert_dos(i) for i in routes_dos]
        data = dfs[0]
        for i in range(self.num):
            data = pd.merge(data, dfs[i])
        return data
    
##################################################
##################################################