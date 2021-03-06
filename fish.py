
## Load required packages
import numpy as np

import itertools,random

from scipy.special import rel_entr, logit
from scipy.stats import mode
from scipy.stats import norm

from sklearn.metrics import auc

import matplotlib.pyplot as plt
from matplotlib import cm



## Define a fish object

naive_escalation = {
      0:.10,
      1:.20,
      2:.30,
      3:.50,
      4:.70
}

## Fish object with internal rules and estimates
class Fish:
    def __init__(self,idx=0,age=50,size=None,
                 prior=None,likelihood=None,likelihood_dict=None,hock_estimate=.5,update_method='bayes',decay=2,decay_all=False,
                 effort_method=[1,1],fight_params=[.3,.3,.1],escalation=naive_escalation,xs=np.linspace(7,100,500)):
        self.idx = idx
        self.name = idx
        self.age = age
        self.xs = xs
        if size is not None:
            if size == 0:   
                self.size = self._growth_func(self.age)
            else:
                self.size = size
        else:
            mean = self._growth_func(self.age)
            sd = mean/5
            self.size = np.random.normal(mean,sd)
        if prior == True:
            self.prior = norm.pdf(self.xs,self.size,self.size/5)
            self.prior = prior / np.sum(prior)
        elif prior is not None:
            self.prior = pior
        else:
            prior = self._prior_size(self.age,xs=self.xs)
            self.prior = prior / np.sum(prior)
        ## Define the rules one when to escalate, based on confidence
        self.escalation_thresholds = escalation
        self.cdf_prior = self._get_cdf_prior(self.prior)
        self.estimate = self.xs[np.argmax(self.prior)]
        prior_mean,prior_std = self.get_stats()
        self.estimate_ = prior_mean
        if hock_estimate == 'estimate':
            self.hock_estimate = self.estimate
        else:
            self.hock_estimate = hock_estimate
        self.hock_record = [self.hock_estimate]
        self.win_record = []
        self.est_record = [self.estimate]
        self.est_record_ = [self.estimate_]
        self.sdest_record = [prior_std]
        self.effort_method = effort_method
        self.decay_all = decay_all
        self.decay = decay
        self.discrete = False
        if update_method == 'bayes':
            #print('using bayes')
            self.update = self.update_prior
        elif update_method == 'hock':
            self.update = self.update_hock
        elif update_method == 'fixed':
            self.update = self._set_boost
            self.discrete = True
        elif update_method == 'decay':
            self.update = self._set_boost
            self.discrete = True
        elif update_method == 'size_boost':
            self.update = self._size_boost
            self.discrete = True
        else:
            #print('setting no update')
            self.update = self.no_update
        self.update_method = update_method
        self.effort = 0
        self.wager = 0
        self.boost = 0 ## Initial boost, needs to be 0, will change with winner/loser effect
        ## Define naive prior/likelihood for 'SA'
        self.naive_params = fight_params
## This should be an average fish.
        naive_prior = self._prior_size(self.age,xs=self.xs)
        self.naive_prior = naive_prior / np.sum(naive_prior)
        self.naive_estimate = np.sum(self.prior * self.xs / np.sum(self.prior))
        if likelihood is not None:
            #print('using existing likelihood')
            self.naive_likelihood = likelihood
        elif self.effort_method[1] == 0:
            self.naive_likelihood = self._define_naive_likelihood()
        else:
            self.naive_likelihood = None
        self.likelihood_dict = likelihood_dict
## Apply winner/loser effect. This could be more nuanced, eventually should be parameterized.
        
    def _set_boost(self,win,fight):
        if win:
            other_fish = fight.loser
            self.boost = np.clip(self.boost + .1,-.5,.5)
        else:
            other_fish = fight.winner
            self.boost = np.clip(self.boost - .1,-.5,.5)
        self.win_record.append([other_fish.size,win,self.effort])
        
    def _size_boost(self,win,fight,shift=2):
        if win:
            other_fish = fight.loser
            self.estimate += shift
            self.estimate_ += shift
        else:
            other_fish = fight.winner
            self.estimate -= shift
            self.estimate_ -= shift
        self.estimate = np.clip(self.estimate,7,100)
        self.win_record.append([other_fish.size,win,self.effort])
        self.est_record.append(self.estimate)

    def _get_cdf_prior(self,prior):
        normed_prior = self.prior / np.sum(self.prior)
        cdf_prior = np.cumsum(normed_prior)
        return cdf_prior

    def _update(self,prior,likelihood,xs = None):
        post = prior * likelihood
        #post = post / auc(xs,post)
        post = post / np.sum(post)
        return post
    
    def _decay_flat(self,prior,xs = None,decay=.001):
        post = prior + np.ones_like(prior) * decay
        post = post / np.sum(post)
        return post

    def _decay_norm(self,prior,xs = None,decay = 2):
        post = prior + (self.naive_prior ** decay)
        post = post / np.sum(post)
        return post

    def _prior_size(self,t,xs = None):
        if xs is None:
            xs = self.xs
        s_mean = self._growth_func(t)
        sd = s_mean / 5
        prior = norm.pdf(xs,s_mean,sd)
        return prior / np.sum(prior)

    def _growth_func(self,t,s_max = 100):
        size = 7 + s_max - s_max/np.exp(t/100)
        return size
    
    def _win_by_ratio(self,r, k=0.1,m=0):
        # sigmoid function
        # use k to adjust the slope
        p = 1 / (1 + np.exp(-(r-m) / k))
        return p

## This is a little clunky to have to define this here also, be sure to know if it matches the fight._wager_curve
## Technically, fight could inherit this from here, but this opens the possibility of having different fish with different LF's. 
    def _wager_curve(self,w,l=.25):
        a = logit(1-l)
        prob_win = w ** (float(np.abs(a))**np.sign(a)) / 2
        return prob_win

## As below, but instead it's based on the wager (assuming opponent size and effort are unknown)
## Assumes opponent size is your own estimate. It's all a mess. 
## NOTE: Come back
    def _likelihood_function_wager(self,x,opp_size=None,opp_effort=None,opp_wager=None,outcome_params = [.3,.3,.3]):
        s,e,l = outcome_params
        if opp_size is None:
            opp_size = self.est_record[0] ## Assume all opponents are average size
        opp_rel_size = opp_size / max([opp_size,x])
        if opp_effort is None:
            if opp_wager is not None:
                opp_effort = (opp_wager ** (1/e)) / (opp_rel_size ** s) ## This should work, but check it
            else:
                opp_effort = opp_size / 100
        if opp_wager is None:
            opp_wager = (opp_rel_size ** s) * (opp_effort ** e)
        rel_size = x / max([opp_size,x])
        my_wager = (rel_size ** s) * (self.effort ** e)
        if my_wager > opp_wager:
            rel_wager = opp_wager / max([my_wager,opp_wager])
            p_win = 1-self._wager_curve(rel_wager,l)
        else:
            rel_wager = my_wager / max([my_wager,opp_wager])
            p_win = self._wager_curve(rel_wager,l)
        return p_win

## It would be nice to just update all this to include fight info
    def _define_likelihood_w(self,outcome_params = [.3,.3,.3],win=True):
        if xs is None:
            xs = self.xs
        likelihood = np.zeros(len(xs))
        if win:
            for s in range(len(xs)):
                likelihood[s] = self._likelihood_function_wager(xs[s],outcome_params=outcome_params)
        elif not win:
            for s in range(len(xs)):
                likelihood[s] = 1-self._likelihood_function_wager(xs[s],outcome_params=outcome_params)
        return likelihood

    def _use_simple_likelihood(self,fight,win):
        if fight.params != self.naive_params:
            #print('rebuilding likelihood')
            self.naive_likelihood = self._define_naive_likelihood(fight)
            self.naive_params = fight.params
        if win:
            likelihood = self.naive_likelihood
        else:
            likelihood = 1 - self.naive_likelihood
        return likelihood

## This is very slow, slow enough that for simulations, I should do it only once
    def _define_naive_likelihood(self,fight=None):
        #print('initializing likelihood')
        if fight is None:
            s,e,l = self.naive_params
        else:
            s,e,l = fight.params
        likelihood = np.zeros(len(self.xs))
## This assumes that all fish are the same age as you
        for i_ in range(len(self.xs)):
            i = self.xs[i_]
            prob_ij = 0
            for j_ in range(len(self.xs)):
                j = self.xs[j_]
                prob_j = self.naive_prior[j_]
                if prob_j != 0:
                    if i > j:
                        i_wager = (i/100)**e
                        j_wager = (j / i)**s * (j/100)**e
                        rel_wager = j_wager / i_wager
                        likelihood_j = 1 - self._wager_curve(rel_wager,l)
                    elif i <= j:
                        i_wager = (i/j)**s * (i/100)**e
                        j_wager = (j/100)**e
                        rel_wager = i_wager / j_wager
                        likelihood_j = self._wager_curve(rel_wager,l)
                    #print(rel_wager)
                    #print(j,likelihood_j,prob_j)
                    prob_ij += likelihood_j * prob_j
                    #print(i,j,rel_wager)
                else:
                    continue
            #print(prob_ij)
            likelihood[i_] = prob_ij
        return likelihood

## Uses fight as input, runs solo, should be one likelihood to rule them all
    def _define_likelihood_solo(self,fight,win):
        likelihood = np.zeros(len(self.xs))
        s,e,l = fight.params
        if win:
            e_self = fight.winner.effort
            w_opp = fight.loser.wager
## Assume fixed, avg effort for all opponents
## NOTE: Somehow you need to transform size to relative size, without knowing opponent size.
            xs_rel = self.xs / self.estimate 
            for s in range(len(self.xs)):
                likelihood[s] = self._likelihood_function_wager(self.xs[s],outcome_params=fight.params)
        elif not win:
            #print('they lost!')
            e_self = fight.loser.effort
            w_opp = fight.winner.wager
            xs_rel = self.xs / max([fight.loser.size,fight.winner.wager / (self.effort**e)])
            for s in range(len(self.xs)):
                likelihood[s] = 1 - self._likelihood_function_wager(self.xs[s],outcome_params=fight.params)
        return likelihood

## Assumes equal effort, which probably isn't quite right. I could assume accurate effort
    def _likelihood_function_size(self,x,x_opp=50,effort=False):
        if x >=x_opp:
            r_diff = (x - x_opp)/x # Will be positive
        elif x_opp > x:
            r_diff = (x - x_opp)/x_opp # Will be negative
        p_win = self._win_by_ratio(r_diff)
        return p_win

## This is closer to the true likelihood, although maybe it should infer relative effort
    def _likelihood_function_se(self,x,x_opp,fight_params=None):
        if fight_params is None:
            s,e,l = self.naive_params
        else:
            s,e,l = fight_params
        x_eff = x/100
        xo_eff = x_opp/100
        if x >=x_opp:
            r_diff = (x-x_opp)/x
            wager = (r_dff ** s * x_eff ** e) / (x ** e)
            p_win = 1-self._wager_curve(wager,l)
        else:
            r_diff = (x - x_opp)/x_opp # Will be negative
            wager = r_dff ** s * x_eff ** e / (xo_eff ** e) 
            p_win = self._wager_curve(wager,l)
        return p_win

    def _use_mutual_likelihood(self,fight,win=True):
        if self.likelihood_dict is None:
            #print('could not find likelihood')
            likelihood = self._define_likelihood_mutual(fight,win) 
        elif fight.winner.idx not in self.likelihood_dict.keys() or fight.loser.idx not in self.likelihood_dict.keys():
            #print('building custom likelihood')
            likelihood = self._define_likelihood_mutual(fight,win) 

        else:
            if win:
                likelihood = self.likelihood_dict[fight.winner.idx,fight.loser.idx]
            else:
                likelihood = 1-self.likelihood_dict[fight.loser.idx,fight.winner.idx]
        return likelihood

    def _define_likelihood_mutual(self,fight,win=True):
        xs = self.xs
        likelihood = np.zeros(len(xs))
        if win:
            other_fish = fight.loser
        else:
            other_fish = fight.winner
        x_opp = other_fish.size
        if win:
            for s in range(len(xs)):
                likelihood[s] = self._likelihood_function_size(xs[s],x_opp)
        elif not win:
            for s in range(len(xs)):
                likelihood[s] = 1-self._likelihood_function_size(xs[s],x_opp)
        return likelihood
    
## This also includes opponent assesment...need to fix that
    def update_prior_old(self,win,x_opp=False,xs=None,w_opp=None,e_self=None,outcome_params=None):
        if xs is None:
            xs = self.xs
        if x_opp == False:
            likelihood = self._define_likelihood_w(outcome_params,win)
        else:
            likelihood = self._define_likelihood(fight,win)
        #self.win_record.append([x_opp,win])
        self.win_record.append([x_opp,win,self.effort])
        self.prior = self._update(self.prior,likelihood,xs)
        self.cdf_prior = self._get_cdf_prior(self.prior)
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
    
## For testing what happens if you just don't update ever
    def no_update(self,win,fight):
        if win:
            other_fish = fight.loser
        else:
            other_fish = fight.winner
        self.win_record.append([other_fish.size,win,self.effort])
        self.est_record.append(self.estimate)
        return self.prior,self.estimate

## A cleaner version so I'm not passing so many arguments
## What if I want the old way though...
    def update_prior(self,win,fight):
        if win:
            other_fish = fight.loser
        else:
            other_fish = fight.winner
        #print('other fish size:',other_fish.size)
        if self.effort_method[1] == 0:

            #print('using simple likes')
            likelihood = self._use_simple_likelihood(fight,win)
            i_estimate = np.argmax(self.xs > self.estimate)
        else:
            #print('using mutual')
            likelihood = self._use_mutual_likelihood(fight,win)

            #likelihood = self._define_likelihood_mutual(fight,win)
        #print('likelihood at own size estimate:',likelihood[np.argmax(self.xs >= self.estimate)])
        self.win_record.append([other_fish.size,win,self.effort])
        pre_prior = self.prior
        self.prior = self._update(self.prior,likelihood,self.xs)
        if self.decay_all:
            print('decaying...')
            self.prior = self._decay_flat(self.prior)
        self.cdf_prior = self._get_cdf_prior(self.prior)
        estimate = self.xs[np.argmax(self.prior)]

        self.estimate_ = np.sum(self.prior * self.xs / np.sum(self.prior))
        prior_mean,prior_std = self.get_stats()
        self.est_record_.append(prior_mean)

        self.sdest_record.append(prior_std)
        
        self.estimate = estimate
        self.est_record.append(estimate)
        
        return self.prior,self.estimate

    def decay_prior(self,store=False):
        print('decaying..')
        pre_prior = self.prior
        self.prior = self._decay_flat(self.prior)
        self.cdf_prior = self._get_cdf_prior(self.prior)
        estimate = self.xs[np.argmax(self.prior)]

        self.estimate_ = np.sum(self.prior * self.xs / np.sum(self.prior))
        if store:
            prior_mean,prior_std = self.get_stats()
            self.est_record_.append(prior_mean)

            self.sdest_record.append(prior_std)
            
            self.estimate = estimate
            self.est_record.append(estimate)
        return self.prior,self.estimate 

    def update_hock(self,win,fight,scale=.1):
        if win:
            other_fish = fight.loser
        else:
            other_fish = fight.winner
        h_opp = other_fish.hock_estimate
        rel_hock = self.hock_estimate / (self.hock_estimate + h_opp)
        estimate = self.hock_estimate + scale * (win-rel_hock)
        if estimate < .001:
            estimate = .001
        self.hock_estimate = estimate
        self.hock_record.append(estimate)
        
    def plot_prior(self,ax=None):
        if ax is None:
            fig,ax = plt.subplots()
        ax.plot(self.xs,self.prior)
        if ax is None:
            fig.show()
    def summary(self,print_me): # print off a summary of the fish
        if print_me:
            print('My number is:',self.idx)
            print('Actual size:',self.size)
            print('Size estimate:',self.estimate)
            print('Win record:',self.win_record)
        return self.name,self.size,self.estimate,self.win_record
    
    ## This needs to be divied up by strategy somehow...
    def choose_effort_discrete(self,f_opp,strategy=None):
        if strategy is None:
            strategy = self.effort_method
        if strategy == [1,0]:
            effort = self.estimate / 100
        elif strategy == [1,1]:
            effort = np.clip(self.estimate / (2 * f_opp.size),0,1)
        elif strategy == [0,1]:
            effort = 1 - f_opp.size / 100
        else:
            effort = 1
        effort = self._boost_effort(effort)
        return effort

    def choose_effort(self,f_opp,strategy=None):
        if self.discrete:
            return self.choose_effort_discrete(f_opp,strategy)
        if strategy is None:
            strategy = self.effort_method
## The latter strategy here is more in keeping with the probabilistic mutual assessment, just against an average fish
        if strategy == [1,0]:
            #effort = self.estimate / 100
            effort = 1 - self.cdf_prior[np.argmax(self.xs > self.naive_estimate)]
        elif strategy == [0,1]:
            effort = 1 - f_opp.size / 100
        elif strategy == [1,1]:
#NOTE: I think cdf_prior is still a bit off, since it's summing to a very large number. 
            ## I think we could do np.sum(self.cdf_prior * f_opp.cdf_prior)
            #print('judging size:',1-self.cdf_prior[np.argmax(self.xs > f_opp.size)])
            #print('self.estimate/100:',self.estimate/100)
            #print('opp estimate/100:',f_opp.size/100)
            #print('opp assessment/100:',1 - f_opp.size /100)
            
            effort = 1 - self.cdf_prior[np.argmax(self.xs > f_opp.size)]
            effort = np.round(effort,4)
            """
            if effort > .5 and self.estimate < f_opp.size:
                import pdb
                pdb.set_trace()
            """
            #effort =  np.sum(self.cdf_prior[self.xs > f_opp.size]) / np.sum(self.cdf_prior)

        elif strategy == 'ma_c': ## This is the continuous version where there is opponent uncertainty
            total_prob = 0
            opp_estimate = self.estimate_opponent(f_opp.size)
            for i in range(len(f_opp.xs)):
                s = f_opp.xs[i]
                total_prop += np.sum(self.cdf_prior[self.xs > s]) * opp_estimate[i]
            effort =  total_prob
        else:
            effort = 1
        effort = self._boost_effort(effort)
        #print('effort post boost:',effort)
        return effort

## Proc effort and decay when you check it
## This also allows for nuanced winner loser effects
    def _boost_effort(self,effort):
        effort = np.clip(effort + self.boost,0,1)
        self.boost = self.boost ** self.decay
        return effort

    ## Function to estimate opponent size, should return a prior distribution of opponent size like own prior_cdf
    def estimate_opponent(self,f_opp):
        return f_opp.cdf_prior
    
    def escalate_(self,x_opp,level=0): # Decide whether to fight (or escalate a fight) against a bigger fish
        #print('Choosing whether to escalate...Estimate,opponent:',self.estimate,x_opp)

        #NOTE: Switch this to a cdf estimate (i.e. odds of winning)
        if self.estimate > x_opp * (level * .2 + 1): ## This is a wierd structure
            return True
        else:
            return False
    
    ## This is just a little wrapper for probably_bigger
    def escalate(self,x_opp,level=0):
        conf_needed = self.escalation_thresholds[level]
        if self.probably_bigger(x_opp,conf_needed):
            return True
        else:
            return False
    # Check whether you believe you are bigger with some required confidence
    def probably_bigger(self,x_opp,conf=.5):
        cutoff = self.xs[np.argmax(self.cdf_prior > (1-conf))] ## Calculate cutoff for a given confidence
        if x_opp < cutoff:
            return True
        else:
            return False
    

    def fight(self,x_opp): # Decide if you win against another fish
        prob_win = self._likelihood_function_size(self.size,x_opp)
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
        
if __name__ == '__main__':
    f1 = Fish() 
