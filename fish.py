
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
                 prior=None,likelihood=None,hock_estimate=.5,
                 effort_method=[1,1],escalation=naive_escalation,xs=np.linspace(5,150,500)):
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
        if prior is not None:
            self.prior = pior
        else:
            self.prior = self._prior_size(self.age,xs=self.xs)
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

    def _get_cdf_prior(self,prior):
        normed_prior = self.prior / np.sum(self.prior)
        cdf_prior = np.cumsum(normed_prior)
        return cdf_prior

    def _update(self,prior,likelihood,xs = None):
        if xs is None:
            xs = self.xs
        post = prior * likelihood
        #post = post / auc(xs,post)
        post = post / np.sum(post)
        return post
    
    def _prior_size(self,t,xs = np.arange(5,150)):
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
    def _likelihood_function_wager(self,x,e_self=1.0,w_opp=.5,outcome_params = [.3,.3,.3]):
        s,e,l = outcome_params
        w_self = (x**s) * (e_self ** e) 
        if w_self>=w_opp:
            r_diff = w_opp/w_self
            p_win = 1 - self._wager_curve(r_diff,l)
        elif w_opp > w_self:
            r_diff = w_self / w_opp 
            p_win = self._wager_curve(r_diff,l)
        return p_win

    def _define_likelihood_w(self,e_self=1.0,w_opp=.5,outcome_params = [.3,.3,.3],xs=None,win=True):
        if xs is None:
            xs = self.xs
        likelihood = np.zeros(len(xs))
        if win:
            for s in range(len(xs)):
                likelihood[s] = self._likelihood_function_wager(xs[s],e_self,w_opp,outcome_params)
        elif not win:
            for s in range(len(xs)):
                likelihood[s] = 1-self._likelihood_function_wager(xs[s],e_self,w_opp,outcome_params)
        return likelihood

   def _likelihood_function_size(self,x,x_opp=50):
        if x >=x_opp:
            r_diff = (x - x_opp)/x # Will be positive
        elif x_opp > x:
            r_diff = (x - x_opp)/x_opp # Will be negative
        p_win = self._win_by_ratio(r_diff)
        return p_win

    def _define_likelihood(self,x_opp=50,xs=None,win=True):
        if xs is None:
            xs = self.xs
        likelihood = np.zeros(len(xs))
        if win:
            for s in range(len(xs)):
                likelihood[s] = self._likelihood_function_size(xs[s],x_opp)
        elif not win:
            for s in range(len(xs)):
                likelihood[s] = 1-self._likelihood_function_size(xs[s],x_opp)
        return likelihood
     
    def update_prior(self,win,x_opp,xs=None):
        if xs is None:
            xs = self.xs
        likelihood = self._define_likelihood(x_opp,xs,win)
        self.win_record.append([x_opp,win])
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
    
    def update_hock(self,win,h_opp,scale=.1):
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
    def summary(self): # print off a summary of the fish
        print('My number is:',self.idx)
        print('Actual size:',self.size)
        print('Size estimate:',self.estimate)
        print('Win record:',self.win_record)
        return self.name,self.size,self.estimate,self.win_record
    
    ## This needs to be divied up by strategy somehow...
    def choose_effort(self,f_opp,strategy=None):
        if strategy is None:
            strategy = self.effort_method
        if strategy == [1,0]:
            return self.estimate / 100 ## This is defined as the max fish size, which could change...
        elif strategy == [0,1]:
            return 1 - f_opp.size / 100
        elif strategy == [1,1]:
#NOTE: I think cdf_prior is still a bit off, since it's summing to a very large number. 
            ## I think we could do np.sum(self.cdf_prior * f_opp.cdf_prior)
            #print('judging size:',np.sum(self.cdf_prior[self.xs > f_opp.size]))
            #print('self.estimate:',self.estimate/100)
            #print('opp assessment:',1 - f_opp.size /100)
            return np.sum(self.cdf_prior[self.xs > f_opp.size])

        elif strategy == 'ma_c': ## This is the continuous version where there is opponent uncertainty
            total_prob = 0
            opp_estimate = self.estimate_opponent(f_opp.size)
            for i in range(len(f_opp.xs)):
                s = f_opp.xs[i]
                total_prop += np.sum(self.cdf_prior[self.xs > s]) * opp_estimate[i]
            return total_prob
        else:
            return 1
    
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
