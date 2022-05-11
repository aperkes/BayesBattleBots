
## Load required packages
import numpy as np

import itertools,random

from scipy.special import rel_entr
from scipy.stats import mode
from scipy.stats import norm

from sklearn.metrics import auc

import matplotlib.pyplot as plt
from matplotlib import cm

## Define simulation class:

class SimParams():
    def __init__(self,n_iterations=1000,n_fish=4,n_rounds=200,fight_selection='random',
                effort_method=[1,1],outcome_params=[.3,.3,.3],update_method='bayes',effect_strength=[1,1]):
        self.n_iterations = n_iterations
        self.n_fish = n_fish
        self.n_rounds = n_rounds
        self.fight_selection = fight_selection ## this defines how much fish can pick their opponents
        self.effort_method = effort_method     ## This is self-assessment vs opponent assessment, [1,1] is MA
        self.outcome_params = outcome_params   ## This determines how fights are settled, skill,effort,luck
        self.update_method = update_method     ## This determines how individuals update their self assessment
        self.effect_strength = effect_strength ## This determines the relative strenght of the winner & loser effects
        
    def summary(self):
        print('Number iterations:',self.n_iterations)
        print('Number of Fish:',self.n_fish)
        print('Number of rounds:',self.n_rounds)
        print('Fight Selection:',self.fight_selection)
        print('Fight Outcome:',self.outcome_params)
        print('Effort Method:',self.effort_method)
        print('Update Method:',self.update_method)

"""    def __init__(self,fishes,fight_list = None,n_fights = None,
                 f_method='balanced',f_outcome='math',f_params=[.3,.3,.3],u_method='bayes'):
"""
        
class Simulation():
    def __init__(self,params=SimParams()):
        self.params = params
        
    def run_simulation(self):
        all_stats = []
        for i in range(self.n_iterations):
            tank = self._build_tank(i)
            tank.run_all()
            t_stats = _get_tank_stats(tank)
            all_stats.append(t_stats)
    
    def _build_tank(self,i): ## I Might want the iteration info available
        p = self.params
        fishes = [Fish() for f in p.n_fish]
        n_fights = p.n_rounds
        
        return Tank(fishes,n_fights=p.n_rounds,f_method=p.fight_selection,f_params=p.outcome_params)
    
    def _get_tank_stats(self,tank):
        return self._calc_linearity(tank),self._calc_stability(tank),self._calc_accuracy(tank)
    
    def _calc_linearity(self,tank):
        n_fish = len(tank.fishes)
        h_matrix = np.zeros([n_fish,n_fish])
        win_record_dif = tank.win_record - np.transpose(tank.win_record)
        h_matrix[win_record_dif > 0] = 1
        tank.h_matrix = h_matrix
        
        ## DO THE MATHY THING HERE
        
        return linearity
        
    def _calc_stability(self,tank):
        ## This means working through tank matrix by time, and I guess it's the standard deviation or something?
        if tank.fight_selection == 'balanced':
            stability = np.mean(np.std(tank.history,axis=0))
        else:
            ## First calculate a sliding window bigger than 2*n^2. We're going to have some missing values
            min_slide = 2*tank.n_fish*(tank.n_fish-1)
            ## will need to test this...
            binned_history = np.convolve(mydata,np.ones([min_slide,tank.n_fish,tank.n_fish],dtype=int),'valid')
            stability = np.mean(np.std(binned_history,axis=0))
        return stability
    
    def _calc_accuracy(self,tank):
        ## This is the correlation between size rank and hierarchy rank
        ## It could also be the coefficient, to be even more precise...
        return 0
    
    def get_timed_means(self):
        ## Maybe this is a bad idea actually...It's really just binned history.
        return linearity_history,stability_history,accuracy_history
    
    def get_mean_stats(self): ## this is calculating these across all tanks
        
        return mean_linearity,mean_stability,mean_accuracy
    
    def get_final_stats(self): ## This is calculating across all thanks, but only using the final info
        
        return final_linearity,final_stability,final_accuracy
