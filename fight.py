
## Load required packages
import numpy as np

import itertools,random

from scipy.special import rel_entr, logit
from scipy.stats import mode
from scipy.stats import norm

from sklearn.metrics import auc

import matplotlib.pyplot as plt
from matplotlib import cm

from fish import Fish

## Simple-ish object to keep track of matchups (and decide outcome if necessary)
class Fight():
    def __init__(self,fish1,fish2,outcome="math",outcome_params=[.5,.5,.1],level=None,scale=.1,idx=0,food=True):
        self.fish1 = fish1
        self.fish2 = fish2
        self.fishes = [fish1,fish2]
        self.mechanism = outcome
        self.level = level
        self.outcome = '?'
        self.params = outcome_params
        self.scale=scale
        self.idx = idx
        self.food = food
        self.p_win = None
        
    def run_outcome(self):
        if self.mechanism == 'chance':
            self.outcome = self.chance_outcome()
        elif self.mechanism == 'escalate':
            self.outcome,self.level = self.escalating_outcome()
        elif self.mechanism == 'escalate_level':
            self.outcome,self.level = self.escalating_by_level()
        elif self.mechanism == 'wager':
            self.outcome,self.level = self.wager()
        elif self.mechanism == 'wager_estimate':
            self.outcome,self.level = self.wager_estimate()
        elif self.mechanism == 'wager_chance':
            self.outcome,self.level = self.wager_chance()
        elif self.mechanism == 'hock':
            scale = self.scale ## Could use this, but have to remember it...
            #scale = .1
            self.outcome,self.level = self.hock_huber(scale=scale,params = self.params)
        elif self.mechanism == 'math':
            #print('using mathy.',self.params)
            self.outcome,self.level = self.mathy(self.params)
            
        else:
            self.outcome = self.mechanism
            self.level = 0.5
      
        self.winner = self.fishes[self.outcome]
        self.loser = self.fishes[1-self.outcome]
        return self.outcome
        
    def chance_outcome(self): ## This just pulls from the fish.fight, basically a dice roll vs the likelihood
        prob_win = likelihood_function_size(self.fish1.size,self.fish2.size)
        self.prob_win = prob_win
        if np.random.random() < prob_win:
            return 0
        else:
            return 1
    
    def escalating_outcome(self):
        cont = True
        level = 0
        focal_fish = np.random.choice([0,1])
        while cont and level < 5:
            focal_fish = 1-focal_fish  ## define the focal fish
            other_fish = 1-focal_fish  ## The other fish is defined as the other fish
            cont = self.fishes[focal_fish].escalate_(self.fishes[other_fish].size)
            if cont == True:
                level += 1
            else:
                return 1-focal_fish,level
        ## IF you escalate to 3, return the winner based on likelihood curve
        winner = self.chance_outcome()
        return winner,level
            
    def escalating_by_level(self):
        cont = True
        level = 0
        focal_fish = np.random.choice([0,1])
        while cont and level < 5:
            focal_fish = 1-focal_fish  ## define the focal fish
            other_fish = 1-focal_fish  ## The other fish is defined as the other fish
            cont = self.fishes[focal_fish].escalate(self.fishes[other_fish].size,level)
            if cont == True:
                level += 1
            else:
                return 1-focal_fish,level
        ## IF you escalate to 3, return the winner based on likelihood curve
        winner = self.chance_outcome()
        return winner,level

    def _wager_curve_old(self,w,l=.25):
        a = logit(1-l)
        prob_win = w ** (float(np.abs(a))**np.sign(a)) / 2
        return prob_win

    def _wager_curve(self,w,l=.05):
        if l == 0:
            prob_win = 0
        else:
            L = np.tan((np.pi - l)/2)
            prob_win = (w**L) / 2
        return prob_win

    ## This is useful because it paramaterizes the relative impact of size, effort, and luck
    def mathy(self,params=[1,.3,.05]):
        #print(params)
        s,e,l = params
        f1_size = self.fish1.size
        f2_size = self.fish2.size
        max_size = max([f1_size,f2_size])
        f1_rel_size = f1_size / max_size
        f2_rel_size = f2_size / max_size

        f1_effort = self.fish1._choose_effort(self.fish2)
        f2_effort = self.fish2._choose_effort(self.fish1)
        self.fish1.effort = f1_effort
        self.fish2.effort = f2_effort
## It might make more sense to multiply this...but I don't think so
        f1_wager = (f1_rel_size ** s) * (f1_effort ** e)
        f2_wager = (f2_rel_size ** s) * (f2_effort ** e)
        if f2_wager > f2_effort:
            pass
            #print('calculation:',f2_rel_size,s,f2_effort,e)
        self.fish1.wager = f1_wager
        self.fish2.wager = f2_wager

## Alternatively:
        ##min_wager = min([f1_wager,f2_wager]) / (f1_wager + f2_wager)
        #f_min = np.argmin([f1_wager,f2_wager]) ## For some reason, this biases it.
        if f1_wager == f2_wager:
            self.p_win = 0.5
            f_min = np.random.randint(2)
            min_normed = 'even'
        else:
            min_wager = min([f1_wager,f2_wager])
            min_normed = min_wager / max([f1_wager,f2_wager])
            f_min = 1 - np.argmax([f1_wager,f2_wager])
            self.p_win = self._wager_curve(min_normed,l)
        roll = random.random()
        if roll < self.p_win: ## probability that the "lower invested" fish wins
            winner = f_min
        else:
            winner = 1-f_min
        loser = 1-winner
        level = min([f1_effort,f2_effort])
        if False and self.fishes[loser].size > self.fishes[winner].size:
            print('#### something is wrong...')
            print(f1_size,f2_size,f1_rel_size,f2_rel_size)
            print(self.p_win,min_normed,winner,f1_wager,f2_wager)
            print('roll:',roll)
        return winner,level
     
    def hock_huber(self,scale=.1,params=[.5,.5,.5]):
        f1_effort = self.fish1.hock_estimate
        f2_effort = self.fish2.hock_estimate
        prob_f1 = f1_effort / (f1_effort + f2_effort)
        f_min = 0
## Alternatively:
        if True:
            s,e,l = params
            f1_size = self.fish1.size
            f2_size = self.fish2.size
            max_size = max([f1_size,f2_size])
            f1_rel_size = f1_size / max_size
            f2_rel_size = f2_size / max_size

            f1_wager = (f1_rel_size ** s) * (f1_effort ** e)
            f2_wager = (f2_rel_size ** s) * (f2_effort ** e)
            min_wager = min([f1_wager,f2_wager]) / max([f1_wager,f2_wager])
            f_min = np.argmin([f1_wager,f2_wager])
            prob_fmin = self._wager_curve(min_wager,l)
            prob_f1 = prob_fmin
        if random.random() < prob_f1:
            winner = f_min 
        else:
            winner = 1-f_min
        level = min([f1_effort,f2_effort]) * scale
        return winner,level
    
    ## Choose based off of estimate
    def wager(self):
        ## Get each fish's estimated probability of being bigger 
        f1_wager = 1 - self.fish1.cdf_prior[np.argmax(self.fish1.xs > self.fish2.size)]
        f2_wager = 1 - self.fish2.cdf_prior[np.argmax(self.fish2.xs > self.fish1.size)]
        if f1_wager > f2_wager:
            winner = 0
            level = f2_wager
        elif f2_wager > f1_wager:
            winner = 1
            level = f1_wager
        elif f2_wager == f1_wager:
            winner = self.chance_outcome()
            level = f1_wager
        return winner, level
    
    def wager_estimate(self):
        f1_wager = self.fish1.estimate
        f2_wager = self.fish2.estimate
        if f1_wager > f2_wager:
            winner = 0
            level = f2_wager
        elif f2_wager > f1_wager:
            winner = 1
            level = f1_wager
        elif f2_wager == f1_wager:
            winner = self.chance_outcome()
            level = f1_wager
        return winner, level
    
    def wager_chance(self):
        f1_wager = self.fish1.estimate
        f2_wager = self.fish2.estimate
        #prob_f1 = f1_wager / (f1_wager + f2_wager)
        prob_f1 = likelihood_function_size(f1_wager,f2_wager)
        if random.random() < prob_f1:
            winner = 0
        else:
            winner = 1
        level = min([f1_wager,f2_wager])
        return winner,level
   
    def summary(self):
        sum_str =  ' '.join([str(self.fish1.idx),'vs',str(self.fish2.idx),str(self.outcome),': So,',str(self.winner.idx),'won, prob of upset was:',str(self.p_win)])
        strat_str = ' '.join(['Fish1:',str(self.fish1.effort_method),'Fish2:',str(self.fish2.effort_method)])
        header_str = 'Fish : Size Own_estimate Opp_estimate Effort'
        effort_str1 = ' '.join(['Fish1:',str(self.fish1.size),str(self.fish1.estimate),str(self.fish1.opp_estimate),str(self.fish1.effort)])
        effort_str2 = ' '.join(['Fish2:',str(self.fish2.size),str(self.fish2.estimate),str(self.fish2.opp_estimate),str(self.fish2.effort)])
        sum_str = '\n'.join([sum_str,strat_str,header_str,effort_str1,effort_str2])
        return sum_str

if __name__ == "__main__":
    f1 = Fish()
    f2 = Fish()
    f = Fight(f1,f2)
