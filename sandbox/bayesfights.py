#! /usr/bin/env python

## Script version of jupyter notebook for bayes winner-effect simulation

## Import stuff
import numpy as np

from math import comb
import itertools,random

from scipy.special import rel_entr
from scipy.stats import mode
from scipy.stats import norm

from sklearn.metrics import auc

import matplotlib.pyplot as plt
from matplotlib import cm

## Prior function


def prior_size(t,xs = np.arange(5,150)):
    s_mean = growth_func(t)
    sd = s_mean / 5
    return norm.pdf(xs,s_mean,sd)

def growth_func(t,s_max = 100):
    size = 7 + s_max - s_max/np.exp(t/100)
    return size

## Likelihood function

def win_by_ratio(r, k=0.1,m=0):
    # sigmoid function
    # use k to adjust the slope
    p = 1 / (1 + np.exp(-(r-m) / k)) 
    return p

def likelihood_function_size(x,x_opp=50):
    if x >=x_opp:
        r_diff = (x - x_opp)/x # Will be positive
    elif x_opp > x:
        r_diff = (x - x_opp)/x_opp # Will be negative
    p_win = win_by_ratio(r_diff)
    return p_win
    
def define_likelihood(x_opp=50,xs=np.arange(5,150),win=True):
    likelihood = np.zeros(len(xs))
    if win:
        for s in range(len(xs)):
            likelihood[s] = likelihood_function_size(xs[s],x_opp)
    elif not win:
        for s in range(len(xs)):
            likelihood[s] = 1-likelihood_function_size(xs[s],x_opp)
    return likelihood


## Define a fish object

class Fish:
    def __init__(self,name=0,age=50,size=None,prior=None,likelihood=None,xs=np.linspace(5,150,500)):
        self.name = name
        self.age = age
        self.xs = xs
        if size is not None:
            if size == 0:   
                self.size = growth_func(self.age)
            else:
                self.size = size
        else:
            mean = growth_func(self.age)
            sd = mean/5
            self.size = np.random.normal(mean,sd)
        if prior is not None:
            self.prior = pior
        else:
            self.prior = self._prior_size(self.age,xs=self.xs)
        
        self.estimate = self.xs[np.argmax(self.prior)]
        prior_mean,prior_std = self.get_stats()
        self.estimate_ = prior_mean
        self.win_record = []
        self.est_record = [self.estimate]
        self.est_record_ = [self.estimate_]
        self.sdest_record = [prior_std]
        
    def _update(self,prior,likelihood,xs = None):
        if xs is None:
            xs = self.xs
        post = prior * likelihood
        post = post / auc(xs,post)
        return post
    
    def _prior_size(self,t,xs = np.arange(5,150)):
        s_mean = self._growth_func(t)
        sd = s_mean / 5
        return norm.pdf(xs,s_mean,sd)

    def _growth_func(self,t,s_max = 100):
        size = 7 + s_max - s_max/np.exp(t/100)
        return size
    
    
    def update_prior(self,win,x_opp,xs=None):
        if xs is None:
            xs = self.xs
        likelihood = define_likelihood(x_opp,xs,win)
        self.win_record.append([x_opp,win])
        self.prior = self._update(self.prior,likelihood,xs)
        if True: ## Need to decide which of these to use...
            estimate = self.xs[np.argmax(self.prior)] ## This is easy, but should it be the mean?
        else:
            estimate = np.sum(self.prior * self.xs / np.sum(self.prior))
        self.estimate_ = np.sum(self.prior * self.xs / np.sum(self.prior))
        
        prior_mean,prior_std = self.get_stats()
        self.est_record_.append(prior_mean)
        self.sdest_record.append(prior_std)
        
        self.estimate = estimate
        self.est_record.append(estimate)
        
        return self.prior,self.estimate
    
    def plot_prior(self,ax=None):
        if ax is None:
            fig,ax = plt.subplots()
        ax.plot(self.xs,self.prior)
        if ax is None:
            fig.show()
    def summary(self): # print off a summary of the fish
        print('My name is:',self.name)
        print('Actual size:',self.size)
        print('Size estimate:',self.estimate)
        print('Win record:',self.win_record)
        return self.name,self.size,self.estimate,self.win_record
    
    def escalate(self,x_opp): # Decide whether to fight (or escalate a fight) against a bigger fish
        #print('Choosing whether to escalate...Estimate,opponent:',self.estimate,x_opp)
        if self.estimate > x_opp:
            return True
        else:
            return False
    
    def fight(self,x_opp): # Decide if you win against another fish
        prob_win = likelihood_function_size(self.size,x_opp)
        roll = np.random.random()
        if roll < prob_win: # This should be right, >= would bias it, < is fair I think
            return True
        else:
            return False
    def fight_fixed(self,x_opp): # Same as above, but the bigger fish always wins
        if self.size > x_opp:
            return True
        else:
            return False
    def get_stats(self):
        prior_mean = np.sum(self.prior * self.xs / np.sum(self.prior))
        prior_std = np.sum((self.xs - prior_mean)**2 * self.prior/(np.sum(self.prior)))
        prior_std = np.sqrt(prior_std)
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
        return prior_mean,prior_std
        
## Define a long-term tank class:

## Fight is a dictionary or maybe class:
#Fish 1 vs Fish 2
#Winner: Fish 2
#Extent: None

## Super simple object to keep track of matchups (and decide outcome if necessary)


class Fight():
    def __init__(self,fish1,fish2,outcome="chance",level=None):
        self.fish1 = fish1
        self.fish2 = fish2
        self.fishes = [fish1,fish2]
        self.mechanism = outcome
        self.level = level
        self.outcome = '?'
        
    def run_outcome(self):
        if self.mechanism == 'chance':
            self.outcome = self.chance_outcome()
        elif self.mechanism == 'escalate':
            self.outcome,self.level = self.escalating_outcome()
        else:
            self.outcome = self.mechanism
      
        self.winner = self.fishes[self.outcome]
        self.loser = self.fishes[1-self.outcome]
        
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
        while cont and level < 3:
            focal_fish = 1-focal_fish  ## define the focal fish
            other_fish = 1-focal_fish  ## The other fish is defined as the other fish
            cont = self.fishes[focal_fish].escalate(self.fishes[other_fish].size)
            if cont == True:
                level += 1
            else:
                return 1-focal_fish,level
        ## IF you escalate to 3, return the winner based on likelihood curve
        winner = self.chance_outcome()
        return winner,level
            
    def summary(self):
        prob_outcome = self.prob_win
        if self.outcome:
            prob_outcome = 1 - prob_outcome
        
        return ' '.join([str(self.fish1.name),'vs',str(self.fish2.name),str(self.outcome),': So,',str(self.winner.name),'won, with probability',str(prob_outcome)])


class Tank():
    def __init__(self,fishes,fight_list = None,n_fights = None,f_method='balanced',f_outcome='chance'):
        self.fishes = fishes
        self.f_method = f_method
        self.f_outcome = f_outcome
        if fight_list is not None:
            self.fight_list = fight_list
        else:
            if n_fights is None:
                ## if n is not defined, just run each one once
                self.fight_list = self.get_matchups(f_method,f_outcome)
                self.n_fights = len(self.fight_list)
            else:
                self.fight_list = self.get_matchups(f_method,f_outcome,n_fights)
            
            ## Randomly match up fishes
    
    def get_matchups(self,f_method='balanced',f_outcome='chance',n_fights=10):
        fight_list = []
        
        if f_method == 'balanced':
            short_list = []
            for i in range(n_fights):
                for f1,f2 in itertools.combinations(self.fishes, 2):
                    fight_list.append(Fight(f1,f2,outcome=f_outcome))
        if f_method == 'random':
            combs = list(itertools.combinations(self.fishes,2))
            for n in range(n_fights):
                f1,f2 = random.choice(combs)
                fight_list.append(Fight(f1,f2,outcome=f_outcome))
        return fight_list
    
    def process_fight(self,fight): ## This works without returning because of how objects work in python
        fight.run_outcome()
        #print('before,after')
        #print(fight.winner.estimate)
        fight.winner.update_prior(True,fight.loser.size)
        #print(fight.winner.estimate)
        #print()
        fight.loser.update_prior(False,fight.winner.size)
        #return fight.winner,fight.loser
    def print_status(self):
        for f in self.fishes:
            print(f.name,':',f.size,f.estimate)
    def run_all(self,print_me=False):
        if print_me:
            print("Starting fights!")
            print("Initial status_____")
            self.print_status()
        for c in self.fight_list:
            self.process_fight(c)
            if print_me:
                print('UPDATE:_____')
                print(c.summary())
                self.print_status()

if __name__ == "__main__":
    f0 = Fish(0,age=50)
    f1 = Fish(1,age=50)

    f2 = Fish(2,age=50)
    f3 = Fish(3,age=50)
    f4 = Fish(4,age=50)

    fish_list = [Fish(f,size=46,age=50) for f in range(10)]

    t0 = Tank(fish_list,f_method = 'random',f_outcome='escalate',n_fights=8000)
    t0.print_status()
    t0.run_all()
    print('Running 1000 times...')
    t0.print_status()

    fig,ax = plt.subplots()

    sizes = []
    for i in range(len(t0.fishes)):
        f = t0.fishes[i]
        sizes.append(f.size)
        ax.plot(f.est_record,color=cm.tab10(i),linestyle=':')
        ax.plot(f.est_record_,color=cm.tab10(i))
        if i == 0:
            ax.axhline(f.size,color =cm.tab10(i),label='Actual Size')
            ax.fill_between(np.arange(len(f.est_record)),np.array(f.est_record_) + np.array(f.sdest_record),
                        np.array(f.est_record_) - np.array(f.sdest_record),
                        color=cm.tab10(i),alpha=.3,label='Estimated Prior')
        else:
            ax.axhline(f.size,color =cm.tab10(i))
            ax.fill_between(np.arange(len(f.est_record)),np.array(f.est_record_) + np.array(f.sdest_record),
                        np.array(f.est_record_) - np.array(f.sdest_record),
                        color=cm.tab10(i),alpha=.3)
        
    ax.axhline(growth_func(50),color='black',label='Original Estimate')
    ax.set_xlabel('n contests')
    ax.set_ylabel('Size (mm)')
    ax.set_ylim([min(sizes)*.8,max(sizes)*1.3])

    fig.legend()
    fig.show()   
