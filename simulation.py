
## Load required packages
import numpy as np

import itertools,random

from scipy.special import rel_entr
from scipy.stats import mode,norm,chi2
from scipy.stats import spearmanr
from scipy.ndimage import convolve
from sklearn.metrics import auc

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm

## Define simulation class:

from fish import Fish
from fight import Fight
from tank import Tank

from elosports.elo import Elo


class SimParams():
    def __init__(self,n_iterations=1000,n_fish=4,n_rounds=200,f_method='random',
                effort_method=[1,1],f_outcome='math',outcome_params=[.3,.3,.3],u_method='bayes',effect_strength=[1,1],scale=.1,verbose=False):
        self.n_iterations = n_iterations
        self.n_fish = n_fish
        self.n_rounds = n_rounds
        self.f_method = f_method ## this defines how much fish can pick their opponents
        self.f_outcome = f_outcome ## This defines how fights are determined.
        self.effort_method = effort_method     ## This is self-assessment vs opponent assessment, [1,1] is MA
        self.outcome_params = outcome_params   ## This determines how fights are settled, skill,effort,luck
        self.u_method = u_method     ## This determines how individuals update their self assessment
        self.effect_strength = effect_strength ## This determines the relative strenght of the winner & loser effects
        self.scale = .1 ## determines scale of hock effect
        self.verbose=verbose
        
    def summary(self):
        print('Number iterations:',self.n_iterations)
        print('Number of Fish:',self.n_fish)
        print('Number of rounds:',self.n_rounds)
        print('Fight Selection Method:',self.f_method)
        print('Fight Outcome:',self.outcome_params)
        print('Effort Method:',self.effort_method)
        print('Update Method:',self.u_method)

class Simulation():
    def __init__(self,params=SimParams()):
        self.params = params
## Frustratingly long table to get p values for stats
        self._applebys = {
            3:{0:0.750},
            4:{0:0.375},
            5:{0:0.117},
            6:{0:0.022,
                1:0.051,
                2:0.120},
            7:{1:0.006,
                2:0.017,
                3:0.033,
                4:0.069,
                5:0.112},
            8:{4:0.006,
                5:0.011,
                6:0.023,
                7:0.037,
                8:0.063,
                9:0.094,
                10:0.153
            },
            9:{9:0.007,
                10:0.012,
                11:0.019,
                12:0.030,
                13:0.045,
                14:0.067,
                15:0.095,
                16:0.138
            },
            10:{16:0.008,
                17:0.012,
                18:0.018,
                19:0.026,
                20:0.038,
                21:0.052,
                22:0.073,
                23:0.097,
                24:0.131
            }
            }

    def run_simulation(self,progress=True):
        all_stats = []
        if self.params.verbose:
            print('Running simulation, n_iterations:',self.params.n_iterations)
        if progress:
            iterator = tqdm(range(self.params.n_iterations))
        else:
            iterator = range(self.params.n_iterations)

        #for i in range(self.params.n_iterations):
        for i in iterator:
            tank = self._build_tank(i)
            if self.params.effort_method[1]==0:
                #print('initializing likelihood')
                tank._initialize_likelihood()
            tank.run_all(False)
            t_stats = self._get_tank_stats(tank)
            all_stats.append(t_stats)
        return all_stats
         
    def _build_tank(self,i): ## I Might want the iteration info available
        p = self.params
        if p.effort_method[1] == 0:
            pilot_fish = Fish(0,effort_method=p.effort_method,update_method=p.u_method,fight_params=p.outcome_params)
            naive_likelihood = pilot_fish.naive_likelihood
        else:
            naive_likelihood = None
        fishes = [Fish(f,likelihood=naive_likelihood,effort_method=p.effort_method,update_method=p.u_method,fight_params=p.outcome_params) for f in range(p.n_fish)]
        n_fights = p.n_rounds
        
        return Tank(fishes,n_fights=p.n_rounds,f_method=p.f_method,f_outcome=p.f_outcome,f_params=p.outcome_params,u_method=p.u_method,scale=p.scale)
    
    def _get_tank_stats(self,tank):
        linearity,(d,p) = self._calc_linearity(tank)
        return p,self._calc_stability(tank),self._calc_accuracy(tank),self._calc_effort(tank)
    
    def _calc_linearity(self,tank,idx=None): ## idx is a slice object
        n_fish = len(tank.fishes)
        h_matrix = np.zeros([n_fish,n_fish])
        if idx is not None:
            win_record = np.nansum(tank.history[idx],axis=0)
        else:
            win_record = tank.win_record
        win_record_dif = win_record - np.transpose(win_record)
        h_matrix[win_record_dif > 0] = 1
        tank.h_matrix = h_matrix
        ## DO THE MATHY THING HERE
        N = n_fish

        D = N * (N-1) * (N-2) / 6 ## Total number of possible triads
## Calculate the number of triads: 
        d = N * (N-1) * (2*N-1) / 12 - 1/2 * np.sum(np.sum(h_matrix,1) ** 2) ## From Appleby, 1983
        if N <= 10:
            if d in self._applebys[N].keys():
                p = self._applebys[N][round(d)]
            elif d < min(self._applebys[N].keys()):
                p = min(self._applebys[N].values())
            else:
                p = max(self._applebys[N].values())
        else:
            df = N*(N-1)*(N-2)/(N-4)**2
            chi_stat = (8/(N-4)) * ((N*(N-1)*(N-2)/24) - d + 0.5) + df
            p = 1 - chi2.cdf(chi_stat,df)
        linearity = 1 - (d / D) ## Percentage of non triadic interactions
        #import pdb;pdb.set_trace()
        return linearity,[d,p]
        
## Stability the proportion of interactions consistent with the overall mean. 
    def _calc_stability(self,tank):
        ## This means working through tank matrix by time, and I guess it's the standard deviation or something?
## A nicer metric would be the proportion of bins where mean heirarchy == overall hierarchy, 
        if tank.f_method == 'balanced':
            binned_history = tank.history
        else: ## Something feels wrong here... 
            ## First calculate a sliding window bigger than 2*n^2. We're going to have some missing values
            min_slide = 2*tank.n_fish*(tank.n_fish-1)
            n_points = len(tank.history)
            stagger = 2 # determines the degree to which windows overlap
            n_bins = int(n_points / min_slide * stagger)
            win_size = int(n_points / n_bins)
            binned_history = np.zeros([n_bins,tank.n_fish,tank.n_fish])
## There might be a more efficient way to do this, but this shoudl work.
            for w in np.arange(n_bins):
                h0 = w*win_size
                h1 = (w+1)*win_size
                binned_history[w] = np.sum(tank.history[h0:h1],0)
            ## This doesn't quite work
            #kernel = np.ones([min_slide,tank.n_fish,tank.n_fish])
            #binned_history = convolve(tank.history,kernel)
# Instead, calculate the proportion of binned interactions that = overall interactions
        binary_bins = np.sign(binned_history - np.transpose(binned_history,axes=[0,2,1]))
        mean_history = np.mean(tank.history,0)
        binary_final = np.sign(mean_history - np.transpose(mean_history))

## Use nCr formulat to get the total number of possible interactions
        total_interactions = len(binary_bins) * tank.n_fish * (tank.n_fish-1) 
        #binary_difference = np.clip(np.abs(binary_bins - binary_final),0,1)
        binary_difference = np.abs(binary_bins - binary_final) == 2
        number_consistent = total_interactions - np.sum(binary_difference)
        proportion_consistent = number_consistent / total_interactions 
        #stability = np.mean(np.std(binned_history,axis=0))
        return proportion_consistent
    
## Calculate dominance using some metric...ELO? 
    def _calc_dominance(self,tank):
        ## There is almost certainly a better package for this...
        eloTank = Elo(k=20)
        for f in tank.fishes: # add all the fishes to the 'league'
            eloTank.addPlayer(f.idx) 
        for c in tank.fight_list: # work through all the fights (slow...)
            if c.outcome is not None:
                eloTank.gameOver(winner = c.winner.idx,loser=c.loser.idx)
        dominance = [eloTank.ratingDict[f.idx] for f in tank.fishes]
        return dominance

    ## This is the spearman correlation between size rank and hierarchy rank
    def _calc_accuracy(self,tank):
        tank.rankings = self._calc_dominance(tank)
        #print('rankings,sizes:')
        #print(np.round(tank.rankings,2)[np.argsort(tank.sizes)],np.round(tank.sizes,2)[np.argsort(tank.sizes)])
        accuracy,_ = spearmanr(tank.sizes,tank.rankings)
        ## It could also be the coefficient, to be even more precise...
        return accuracy
    
    def _calc_est_accuracy(self,tank):
        tank.estimates = [f.estimate for f in tank.fishes]
        accuracy,_ = spearmanr(tank.sizes,tank.estimates)
        return accuracy

    def _calc_effort(self,tank):
        effort_record = [f.level for f in tank.fight_list]
        return np.mean(effort_record)

    def get_timed_means(self):
        ## Maybe this is a bad idea actually...It's really just binned history.
        return linearity_history,stability_history,accuracy_history
    
    def get_mean_stats(self): ## this is calculating these across all tanks
        
        return mean_linearity,mean_stability,mean_accuracy
    
    def get_final_stats(self): ## This is calculating across all thanks, but only using the final info
        
        return final_linearity,final_stability,final_accuracy

_applebys = {
    3:{0:0.750},
    4:{0:0.375},
    5:{0:0.117},
    6:{0:0.022,
        1:0.051,
        2:0.120},
    7:{1:0.006,
        2:0.017,
        3:0.033,
        4:0.069,
        5:0.112},
    8:{4:0.006,
        5:0.011,
        6:0.023,
        7:0.037,
        8:0.063,
        9:0.094,
        10:0.153
    },
    9:{9:0.007,
        10:0.012,
        11:0.019,
        12:0.030,
        13:0.045,
        14:0.067,
        15:0.095,
        16:0.138
    },
    10:{16:0.008,
        17:0.012,
        18:0.018,
        19:0.026,
        20:0.038,
        21:0.052,
        22:0.073,
        23:0.097,
        24:0.131
    }
    }

def calc_linearity(tank,idx=None): ## idx is a slice object
    n_fish = len(tank.fishes)
    h_matrix = np.zeros([n_fish,n_fish])
    if idx is not None:
        win_record = np.nansum(tank.history[idx],axis=0)
    else:
        win_record = tank.win_record
    win_record_dif = win_record - np.transpose(win_record)
    h_matrix[win_record_dif > 0] = 1
    tank.h_matrix = h_matrix
    ## DO THE MATHY THING HERE
    N = n_fish

    D = N * (N-1) * (N-2) / 6 ## Total number of possible triads
## Calculate the number of triads: 
    d = N * (N-1) * (2*N-1) / 12 - 1/2 * np.sum(np.sum(h_matrix,1) ** 2) ## From Appleby, 1983
    if N <= 10:
        if d in _applebys[N].keys():
            p = _applebys[N][round(d)]
        elif d < min(_applebys[N].keys()):
            p = min(_applebys[N].values())
        else:
            p = max(_applebys[N].values())
    else:
        df = N*(N-1)*(N-2)/(N-4)**2
        chi_stat = (8/(N-4)) * ((N*(N-1)*(N-2)/24) - d + 0.5) + df
        p = 1 - chi2.cdf(chi_stat,df)
    linearity = 1 - (d / D) ## Percentage of non triadic interactions
    #import pdb;pdb.set_trace()
    return linearity,[d,p]
 
if __name__ == "__main__":
    params = SimParams(n_iterations=300)
    s = 0
    e = 1
    l = 0
    params.outcome_params = [s,e,l]
    params.effort_method = [1,1]
    params.n_fights = 10*50
    params.n_iterations = 50
    params.n_fish = 7
    params.f_method = 'random' 
    params.f_outcome = 'hock'
    params.u_method = 'hock'

    s = Simulation(params)
    all_stats = s.run_simulation()
    print(all_stats)
    print(np.mean(all_stats,axis=0))
