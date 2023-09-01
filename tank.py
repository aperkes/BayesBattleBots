
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

from fish import Fish,FishNPC
from fight import Fight
from params import Params

class Tank():
    def __init__(self,fishes,params=None,fight_list = None,npc_params=None
                 #n_fights = 10,
                 #f_method='balanced',f_outcome='math',f_params=[.3,.3,.3],
                 #effort_method=[1,1],u_method='bayes',scale=.1,fitness_ratio=None,death=False,food=1
                 ):
        if params is None:
            params = Params()
        self.params = params.copy()
        self.npc_params = npc_params
        self.fishes = fishes
        self.n_fish = len(fishes)
        self.sizes = [f.size for f in fishes]
        self.f_method = params.f_method
        self.f_outcome = params.f_outcome
        self.f_params = params.outcome_params
        if not self.params.L_set:
            self.params.set_L() ## confirm that the L is right
            for f in fishes:
                if not f.params.L_set:
                    print('Warning!! Tank just set L and fish Ls are not set!!!')
                    break
        self.u_method = params.update_method
        #self.scale = params.scale
        self.win_record = np.zeros([len(fishes),len(fishes)])
        self.fitness_ratio=params.fitness_ratio
        self.death = params.death
        self.food = params.food
        if fight_list is not None:
            self.fight_list = fight_list
        elif npc_params is not None:
            self.npc_params = npc_params
            self.fight_list = self.get_npc_fights(npc_params.n_npcs,params.n_fights)
        else:
            self.fight_list = self.get_matchups(self.f_method,self.f_outcome,params.n_rounds)

            #if n_fights is None:
                ## if n is not defined, just run each one once
            #    self.fight_list = self.get_matchups(self.f_method,self.f_outcome,scale=self.scale)
            #else:
            #    self.fight_list = self.get_matchups(f_method,f_outcome,n_fights)
            self.n_fights = len(self.fight_list)
        if self.f_method == 'balanced' or self.f_method == 'shuffled':
            #self.n_rounds = int(len(self.fight_list) / (self.n_fish * (self.n_fish-1) / 2))
            self.n_rounds = self.fight_list[-1].idx+1
        else:
            self.n_rounds = len(self.fight_list)
        self.history = np.zeros([self.n_rounds,len(fishes),len(fishes)])
        #self.history.fill(np.nan)


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
                fight=Fight(self.fishes[i],self.fishes[j],self.params,outcome=0)
                fight.winner,fight.loser = self.fishes[i],self.fishes[j]
                likelihood_dict[i,j] = self.fishes[i]._define_likelihood_mutual(fight)
        for f in self.fishes:
            f.likelihood_dict = likelihood_dict

    def get_npc_fights(self,n_npcs=0,n_fights=10):
        fight_list = []

        if n_npcs == 0:
            n_npcs = len(self.fishes)
        if self.npc_params.real_fish:
            self.npcs = [Fish(n,self.npc_params) for n in range(n_npcs)]
        else:
            self.npcs = [FishNPC(n,self.npc_params) for n in range(n_npcs)]
        npc_sizes = [n.size for n in self.npcs]
        fish_sizes = [f.size for f in self.fishes]
        #print('NPC vs Fish',np.mean(npc_sizes),np.mean(fish_sizes))
        for i in range(n_fights):
            for n in self.npcs:
                for f in self.fishes:
                    fight_list.append(Fight(f,n,self.params,idx=i))
        random.shuffle(fight_list)
        if self.fitness_ratio is not None:
            for i in range(0,len(fight_list),int(1/self.fitness_ratio)):
                fight_list[i].food = False
        return fight_list

    def get_matchups(self,f_method='shuffled',f_outcome='chance',n_fights=10):
## N fights for balanced and shuffled is actually the number of rounds
        fight_list = []
        #import pdb;pdb.set_trace()
        if f_method == 'balanced' or f_method == 'shuffled':
            short_list = []
            for i in range(n_fights):
                combs = list(itertools.combinations(self.fishes,2))
                if f_method == 'shuffled':
                    random.shuffle(combs)
                for f1,f2 in combs:
                    fight_list.append(Fight(f1,f2,self.params,idx=i)) ## So balanced is organized as rounds
        if f_method == 'random':
            combs = list(itertools.combinations(self.fishes,2))
            for i in range(n_fights):
                f1,f2 = random.choice(combs)
                fight_list.append(Fight(f1,f2,self.params,idx=i))
        if self.fitness_ratio is not None:
            for i in range(0,len(fight_list),int(1/self.fitness_ratio)):
                fight_list[i].food = False
        #print('n_fights:',len(fight_list))
        return fight_list

    def process_fight(self,fight): ## This works without returning because of how objects work in python
        fight.run_outcome()
        #print('/nbefore:',fight.winner.energy,fight.loser.energy)
        fight.winner.update(True,fight)
        fight.loser.update(False,fight)

        #print('after:',fight.winner.energy,fight.loser.energy)
        #return fight.winner,fight.loser
        self.win_record[fight.winner.idx,fight.loser.idx] += 1
        self.history[fight.idx,fight.winner.idx,fight.loser.idx] = 1 ## Note, this works a bit different for 'random' and 'balanced'
        #self.history[fight.idx,fight.loser.idx,fight.winner.idx] = 0 ## Note, this works a bit different for 'random' and 'balanced'
        return 0

    def process_hock(self,fight):
        fight.run_outcome()
        fight.winner.update_hock(True,fight,fight.level)
        fight.loser.update_hock(False,fight,fight.level)
        self.win_record[fight.winner.idx,fight.loser.idx] += 1
        self.history[fight.idx,fight.winner.idx,fight.loser.idx] = 1 ## Note, this works a bit different for 'random' and 'balanced'

    def print_status(self):
        for f in self.fishes:
            print(f.name,':','size=',np.round(f.size,3),'estimate=',np.round(f.estimate,3))

    def run_all(self,progress=True,print_me=False,plot_stuff=False,cutoff=None):
        if cutoff is None:
            cutoff = len(self.fight_list)
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
            iterator = tqdm(range(cutoff))
        else:
            iterator = range(cutoff)
        for i in iterator:
            c = self.fight_list[i]
            if self.death:
                if not c.fish1.alive or not c.fish2.alive:
                    #if c.fish1.effort_method == 'Perfect' or c.fish2.effort_method == 'Perfect':
                    #    pass
                    if c.fish1.alive:
                        c.outcome = 0 
                        c.level = 0
                        c.winner = c.fish1
                        c.loser=c.fish2
                        c.fish1.no_update(True,c)
                    elif c.fish2.alive: 
                        c.outcome = 1
                        c.level = 0
                        c.winner = c.fish2,
                        c.loser=c.fish1
                        c.fish2.no_update(True,c)
                    else:
                        c.outcome = None
                    continue
            process(c)
            if False:
                if c.loser.params.mutant is True and c.loser.effort > 0:
                    print('Loser Mutants!')
                    import pdb
                    pdb.set_trace()
            if plot_stuff:
                if c.fish1.idx == 0:
                    ax.plot(c.fish1.naive_likelihood,alpha=.2)

            if print_me:
                print('UPDATE:_____')
                print(c.summary())
                #self.print_status()
        if plot_stuff:
            return fig,ax

    def plot_estimates(self,fish_list=None,food=False):
        fig,ax = plt.subplots()
        if fish_list is None:
            fish_list = self.fishes
        if self.u_method == 'hock':
            for i in range(len(fish_list)):
                f = fish_list[i]
                ax.plot(f.hock_record,color=cm.tab10(i),label=str(i))
                ax.axhline(f.size/100,color=cm.tab10(i),linestyle=':')
            ax.set_ylabel('Hock Estimate')
        else:
            for i in range(len(fish_list)):
                if len(fish_list) > 10:
                    color = cm.viridis(i/len(fish_list))
                else:
                    color = cm.tab10(i)
                f = fish_list[i]
                ax.plot(f.est_record, color=color,label=str(i))
                if food:
                    ax.plot(f.size_record,color=color)
                else:
                    ax.axhline(f.size,color=color,linestyle=':')
                if self.u_method == 'bayes':
                    #ax.fill_between(np.arange(len(f.est_record_)),np.array(f.est_record_) + np.array(f.sdest_record),
                    #    np.array(f.est_record_) - np.array(f.sdest_record),color=color,alpha=.3)
                    f.range_record = np.array(f.range_record)
                    #ax.fill_between(np.arange(len(f.range_record)),f.range_record[:,0],f.range_record[:,3],color=color,alpha=0.1)
                    ax.fill_between(np.arange(len(f.range_record)),f.range_record[:,1],f.range_record[:,2],color=color,alpha=0.2)
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
            if len(fish_list) > 10:
                color= cm.viridis(i/len(fish_list))
            else:
                color=cm.tab10(i)
            f = fish_list[i]
            effort_record = np.array(f.win_record)[:,-1] ## note that there are fish win records and tank win records. yikes.
            smooth_effort = effort_record
            #smooth_effort = gaussian_filter1d(effort_record,3)
            ax.plot(smooth_effort,color=color,alpha=.5,label=str(i))
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
    t.run_all()
    print(np.shape(t.win_record))
