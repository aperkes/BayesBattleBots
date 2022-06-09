
## Load required packages
import numpy as np

import itertools,random

from scipy.special import rel_entr
from scipy.stats import mode
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d 
from sklearn.metrics import auc

import matplotlib.pyplot as plt
from matplotlib import cm

from tqdm import tqdm

from fish import Fish
from fight import Fight

class Tank():
    def __init__(self,fishes,fight_list = None,n_fights = None,
                 f_method='balanced',f_outcome='math',f_params=[.3,.3,.3],
                 effort_method=[1,1],u_method='bayes',scale=.1):
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


## Define a likelihood dict
## Input is focal fish vs other fish, output is likelihood of focal fish *win*
    def _initialize_likelihood(self):
        #print('initializing likelihood for all possible match-ups ')
        likelihood_dict = {}
        for i in range(len(self.fishes)):
            for j in range(len(self.fishes)):
                max_idx = np.argmax([self.fishes[i].size,self.fishes[j].size])
                min_idx = 1-max_idx
                max_size = self.fishes[max_idx].size
                min_size = self.fishes[min_idx].size
## Frustratingly, we have to queue up a little dummy fight here
                fight=Fight(self.fishes[i],self.fishes[j],outcome=0,outcome_params=self.f_params)
                fight.winner,fight.loser = self.fishes[i],self.fishes[j]
                likelihood_dict[i,j] = self.fishes[i]._define_likelihood_mutual(fight)
        for f in self.fishes:
            f.likelihood_dict = likelihood_dict

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
        #print('n_fights:',len(fight_list))
        return fight_list

    def process_fight(self,fight): ## This works without returning because of how objects work in python
        fight.run_outcome()
        #print('before,after')
        #print(fight.winner.estimate)

## NOTE: Eventually should be able to delete all this, but for now let's jsut comment it
        """
        if False:
            if fight.winner.effort_method[1] == 1:
                fight.winner.update_prior(True,fight,fight.loser.size)
                fight.loser.update_prior(False,fight,fight.winner.size)
            #NOTE: uncomment these two lines to try the fancy prior.
            else:
                fight.winner.update_prior(True,fight)
                fight.loser.update_prior(False,fight)
        else:
        """
        fight.winner.update(True,fight)
        fight.loser.update(False,fight)
        #print(fight.winner.estimate)
        #print()

        #return fight.winner,fight.loser
        self.win_record[fight.winner.idx,fight.loser.idx] += 1
        self.history[fight.idx,fight.winner.idx,fight.loser.idx] = 1 ## Note, this works a bit different for 'random' and 'balanced'

    def process_hock(self,fight):
        fight.run_outcome()
        fight.winner.update_hock(True,fight,fight.level)
        fight.loser.update_hock(False,fight,fight.level)
        self.win_record[fight.winner.idx,fight.loser.idx] += 1
        self.history[fight.idx,fight.winner.idx,fight.loser.idx] = 1 ## Note, this works a bit different for 'random' and 'balanced'

    def print_status(self):
        for f in self.fishes:
            print(f.name,':','size=',np.round(f.size,3),'estimate=',np.round(f.estimate,3))
    def run_all(self,progress=True,print_me=False,plot_stuff=False):
        if plot_stuff:
            fig,ax = plt.subplots()
        if print_me:
            print("Starting fights!")
            print("Initial status_____")
            self.print_status()
        if self.u_method == 'hock':
            process = self.process_hock
        else:
            process = self.process_fight
        if progress:
            iterator = tqdm(range(len(self.fight_list)))
        else:
            iterator = range(len(self.fight_list))
        for i in iterator:
            c = self.fight_list[i]
            process(c)
            if plot_stuff:
                if c.fish1.idx == 0:
                    ax.plot(c.fish1.naive_likelihood,alpha=.2)

            if print_me:
                print('UPDATE:_____')
                print(c.summary())
                self.print_status()
        if plot_stuff:
            return fig,ax

    def plot_estimates(self,fish_list=None):
        fig,ax = plt.subplots()
        if fish_list is None:
            fish_list = self.fishes
        if self.u_method == 'hock':
            for i in range(len(fish_list)):
                f = fish_list[i]
                ax.plot(f.hock_record,color=cm.tab10(i),label=str(i))
                ax.axhline(f.size/100,color=cm.tab10(i))
            ax.set_ylabel('Hock Estimate')
        else:
            for i in range(len(fish_list)):
                f = fish_list[i]
                ax.plot(f.est_record, color=cm.tab10(i),label=str(i))
                ax.axhline(f.size,color=cm.tab10(i))
                if self.u_method == 'bayes':
                    ax.fill_between(np.arange(len(f.est_record)),np.array(f.est_record_) + np.array(f.sdest_record),
                        np.array(f.est_record_) - np.array(f.sdest_record),color=cm.tab10(i),alpha=.3)
            ax.set_ylabel('Estimate')
        ax.set_xlabel('contest number')
        ax.legend()
        fig.show()
        return fig,ax

## Similar to above, but here simply plot effort
    def plot_effort(self,fish_list=None):
        fig,ax = plt.subplots()
        if fish_list is None:
            fish_list = self.fishes
        for i in range(len(fish_list)):
            f = fish_list[i]
            effort_record = np.array(f.win_record)[:,2]
            smooth_effort = gaussian_filter1d(effort_record,5)
            ax.plot(smooth_effort,color=cm.tab10(i),alpha=.5,label=str(i))
        ax.legend()
        fig.show()
        return fig,ax

    ## Function to calculate winner effect. s allows you to calculate n-steps into the future
    def calc_winner_effect(self,s=1):
        we_by_fish = []
        le_by_fish = []
        mean_by_fish = []
        size_by_fish = []
        for f in self.fishes:
            win_record = np.array(f.win_record)
            record_post_win = win_record[s:][win_record[0:-s,1] == 1]
            record_post_loss = win_record[s:][win_record[0:-s,1] == 0]
            mean_record = np.mean(win_record[:,1])
            mean_post_win = np.mean(record_post_win)
            mean_post_loss = np.mean(record_post_loss)
            winner_effect = mean_post_win / mean_record
            loser_effect = mean_post_loss / (1-mean_record)
            we_by_fish.append(winner_effect)
            le_by_fish.append(loser_effect)
            mean_by_fish.append(mean_record)
            size_by_fish.append(f.size)
        return we_by_fish,le_by_fish,mean_by_fish,size_by_fish

    def __getitem__(self,idx):
        return self.fishes[idx]
    
if __name__ == '__main__':
    fishes = [Fish(),Fish()]
    t = Tank(fishes)
