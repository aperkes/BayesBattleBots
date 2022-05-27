
## Load required packages
import numpy as np

import itertools,random

from scipy.special import rel_entr
from scipy.stats import mode
from scipy.stats import norm

from sklearn.metrics import auc

import matplotlib.pyplot as plt
from matplotlib import cm

from tqdm import tqdm

from fish import Fish
from fight import Fight

class Tank():
    def __init__(self,fishes,fight_list = None,n_fights = None,
                 f_method='balanced',f_outcome='math',f_params=[.3,.3,.3],u_method='bayes',scale=.1):
        self.fishes = fishes
        self.n_fish = len(fishes)
        self.sizes = [f.size for f in fishes]
        self.f_method = f_method
        self.f_outcome = f_outcome
        self.f_params = f_params
        self.u_method = u_method
        self.scale = scale
        self.win_record = np.zeros([len(fishes),len(fishes)])

        if fight_list is not None:
            self.fight_list = fight_list
        else:
            if n_fights is None:
                ## if n is not defined, just run each one once
                self.fight_list = self.get_matchups(f_method,f_outcome,scale=scale)
            else:
                self.fight_list = self.get_matchups(f_method,f_outcome,n_fights)
            self.n_fights = len(self.fight_list)
        if f_method == 'balanced':
            self.n_rounds = int(len(self.fight_list) / (self.n_fish * (self.n_fish-1) / 2))
        else:
            self.n_rounds = len(self.fight_list)
        self.history = np.zeros([self.n_rounds,len(fishes),len(fishes)])


    ## Randomly match up fishes

    def get_matchups(self,f_method='balanced',f_outcome='chance',n_fights=10,scale=.1):
        fight_list = []

        if f_method == 'balanced':
            short_list = []
            for i in range(n_fights):
                for f1,f2 in itertools.combinations(self.fishes, 2):
                    fight_list.append(Fight(f1,f2,outcome=f_outcome,outcome_params=self.f_params,scale=scale,idx=i)) ## So balanced is organized as rounds
        if f_method == 'random':
            combs = list(itertools.combinations(self.fishes,2))
            for i in range(n_fights):
                f1,f2 = random.choice(combs)
                fight_list.append(Fight(f1,f2,outcome=f_outcome,outcome_params=self.f_params,scale=scale,idx=i))
        print('n_fights:',len(fight_list))
        return fight_list

    def process_fight(self,fight): ## This works without returning because of how objects work in python
        fight.run_outcome()
        #print('before,after')
        #print(fight.winner.estimate)
        if fight.winner.effort_method[1] == 1:
            fight.winner.update_prior(True,fight.loser.size)
            fight.loser.update_prior(False,fight.winner.size)
        #NOTE: uncomment these two lines to try the fancy prior.
        else:
            fight.winner.update_prior_(True,fight)
            fight.loser.update_prior_(False,fight)
        #print(fight.winner.estimate)
        #print()

        #return fight.winner,fight.loser
        self.win_record[fight.winner.idx,fight.loser.idx] += 1
        self.history[fight.idx,fight.winner.idx,fight.loser.idx] = 1 ## Note, this works a bit different for 'random' and 'balanced'

    def process_hock(self,fight):
        fight.run_outcome()
        fight.winner.update_hock(True,fight.loser.hock_estimate,fight.level)
        fight.loser.update_hock(False,fight.winner.hock_estimate,fight.level)
        self.win_record[fight.winner.idx,fight.loser.idx] += 1
        self.history[fight.idx,fight.winner.idx,fight.loser.idx] = 1 ## Note, this works a bit different for 'random' and 'balanced'

    def print_status(self):
        for f in self.fishes:
            print(f.name,':','size=',np.round(f.size,3),'estimate=',np.round(f.estimate,3))
    def run_all(self,print_me=False):
        if print_me:
            print("Starting fights!")
            print("Initial status_____")
            self.print_status()
        if self.u_method == 'hock':
            process = self.process_hock
        else:
            process = self.process_fight
        for i in tqdm(range(len(self.fight_list))):
            c = self.fight_list[i]
            process(c)
            if print_me:
                print('UPDATE:_____')
                print(c.summary())
                self.print_status()

    def plot_estimates(self):
        fig,ax = plt.subplots()
        if self.u_method == 'bayes':
            for i in range(len(self.fishes)):
                f = self.fishes[i]
                ax.plot(f.est_record, color=cm.tab10(i))
                ax.axhline(f.size,color=cm.tab10(i))
                ax.fill_between(np.arange(len(f.est_record)),np.array(f.est_record_) + np.array(f.sdest_record),
                    np.array(f.est_record_) - np.array(f.sdest_record),color=cm.tab10(i),alpha=.3)
            ax.set_ylabel('Estimate')
        elif self.u_method == 'hock':
            for i in range(len(self.fishes)):
                f = self.fishes[i]
                ax.plot(f.hock_record,color=cm.tab10(i))
                ax.axhline(f.size/100,color=cm.tab10(i))
            ax.set_ylabel('Hock Estimate')
        ax.set_xlabel('contest number')
        fig.show()
        return fig,ax

    def __getitem__(self,idx):
        return self.fishes[idx]

if __name__ == '__main__':
    fishes = [Fish(),Fish()]
    t = Tank(fishes)
