
## Load required packages
import numpy as np

import itertools,random

from scipy.special import rel_entr, logit
from scipy.stats import mode
from scipy.stats import norm

from sklearn.metrics import auc

import matplotlib.pyplot as plt
from matplotlib import cm

from params import Params

## Define a fish object

naive_escalation = {
      0:.10,
      1:.20,
      2:.30,
      3:.50,
      4:.70
}

## Fish object with internal rules and estimates
# Reminder: r_rhp is the rate that rhp increases when they eat/win
#           a_growth is whether the growth rate should be asymptotic 
#           Prior is either None, in which case it uses the baseline, 
#               True: in which is uses a true mean with mean/5 as the std
#               an int: in which case it uses the int as the std (10 would be average)
class Fish:
    def __init__(self,idx=0,params=None,
                 #age=50,size=None,
                 prior=None,likelihood=None,likelihood_dict=None,
#hock_estimate=.5,update_method='bayes',decay=2,decay_all=False,
                 #effort_method=[1,1],fight_params=[.6,.3,.01],escalation=naive_escalation,xs=np.linspace(7,100,500),
                 #r_rhp=0,a_growth=True,c_aversion=1,max_energy=1,acuity=10,awareness=10,insight=False,energy_cost=False
                 ):
        self.idx = idx
        self.name = idx

        if params is None:
            params = Params()
## Over rule params if you want to do that.
        if prior is not None:
            params.prior = prior
        if likelihood is not None:
            params.likelihood = likelihood
        if likelihood_dict is not None:
            params.likelihood_dict = likelihood_dict
        self.age = params.age
        self.xs = params.xs
        self.r_rhp = params.r_rhp
        self.a_growth = params.a_growth
        self.c_aversion = params.c_aversion
        self.acuity = params.acuity
        self.awareness = params.awareness
        self.insight = params.insight
        if params.size is not None:
            if size == 0:   
                self.size = self._growth_func(self.age)
            else:
                self.size = params.size
        else:
            mean = self._growth_func(self.age)
            sd = mean/5
            self.size = np.random.normal(mean,sd)
        if params.prior == True:
            self.prior = norm.pdf(self.xs,self.size,self.size/5)
            self.prior = self.prior / np.sum(self.prior)
        elif isinstance(params.prior,int):
            self.estimate = np.clip(np.random.normal(self.size,self.awareness),7,100)
            if params.prior == -1:
                self.prior = np.ones_like(self.xs) / len(self.xs)
            else:
                self.prior = norm.pdf(self.xs,self.estimate,params.prior)
                self.prior = self.prior / np.sum(self.prior)
        elif params.prior is not None:
            self.prior = params.pior
        else:
            prior = self._prior_size(self.age,xs=self.xs)
            self.prior = prior / np.sum(prior)
        ## Define the rules one when to escalate, based on confidence
        #self.escalation_thresholds = escalation
        self.cdf_prior = self._get_cdf_prior(self.prior)
        est_5,est_25,est_75,est_95 = self._get_range_prior(self.cdf_prior)
        self.estimate = self.xs[np.argmax(self.prior)]
        prior_mean,prior_std = self.get_stats()
        self.estimate_ = prior_mean
        #if hock_estimate == 'estimate':
        #    self.hock_estimate = self.estimate
        #else:
        #    self.hock_estimate = hock_estimate
        #self.hock_record = [self.hock_estimate]
        self.win_record = []
        self.est_record = [self.estimate]
        self.est_record_ = [self.estimate_]
        self.sdest_record = [prior_std]
        self.range_record = [[est_5,est_25,est_75,est_95]]
        self.effort_method = params.effort_method
        self.effort = None
        self.decay = 1 ## Deprecated decay
        self.decay_all = False
        self.discrete = False
        update_method = params.update_method
        effort_method = params.effort_method
        if update_method == 'bayes':
            #print('using bayes')
            self.update = self.update_prior
        #elif update_method == 'hock':
        #    self.update = self.update_hock
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

        #print('setting effort...',effort_method)
        if effort_method[0] is None:
            if effort_method[1] is None:
                print('leroy!')
                self._choose_effort = self.leroy_jenkins
            elif effort_method[1] == 0.5: 
                #print('half leroy!')
                self._choose_effort = self.half_jenkins
            elif effort_method[1] == '?':
                self._choose_effort = self.random_effort
            elif effort_method[1] == '!':
                self._choose_effort = self.explore_effort
                self.effort_method = [1,1]
            else:
                #print('continuous leroy')
                self.effort = max_energy * effort_method[1]
                self._choose_effort = self.float_jenkins
        else:
            self._choose_effort = self.choose_effort_energy

        self.update_method = update_method
        #self.effort = 0
        self.wager = 0
        self.boost = 0 ## Initial boost, needs to be 0, will change with winner/loser effect
        self.energy = params.start_energy ## add some cost to competing
        self.max_energy = params.max_energy
        self.energy_cost = params.energy_cost

## Initialize size and energy records
        self.size_record = [self.size]
        self.energy_record = [self.energy]
        self.fitness_record = [0]

        self.alive = True
        self.s_max = 100
        ## Define naive prior/likelihood for 'SA'
        self.naive_params = params.outcome_params
## This should be an average fish.
        naive_prior = self._prior_size(self.age,xs=self.xs)
        self.naive_prior = naive_prior / np.sum(naive_prior)
        self.naive_estimate = np.sum(self.prior * self.xs / np.sum(self.prior))
        if params.likelihood is not None:
            #print('using existing likelihood')
            self.naive_likelihood = params.likelihood
        if self.effort_method[1] == 0:
            self.naive_likelihood = self._define_naive_likelihood()
        else:
            self.naive_likelihood = None
        self.likelihood_dict = params.likelihood_dict
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
    
    def _get_range_prior(self,cdf_prior):
        cdf_5 = np.argmax(cdf_prior > 0.05)
        cdf_25 = np.argmax(cdf_prior > 0.25)
        cdf_75 = np.argmax(cdf_prior > 0.75)
        cdf_95 = np.argmax(cdf_prior >= 0.95)
        return self.xs[cdf_5],self.xs[cdf_25],self.xs[cdf_75],self.xs[cdf_95]

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
            rel_wager = opp_wager / my_wager
            p_win = 1-self._wager_curve(rel_wager,l)
        elif my_wager == opp_wager:
            p_win = 0.5
        else:
            rel_wager = my_wager / opp_wager
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
        if fight.outcome_params != self.naive_params:
            #print('rebuilding likelihood')
            self.naive_likelihood = self._define_naive_likelihood(fight)
            self.naive_params = fight.outcome_params
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
            s,e,l = fight.outcome_params
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
        s,e,l = fight.outcome_params
        if win:
            e_self = fight.winner.effort
            w_opp = fight.loser.wager
## Assume fixed, avg effort for all opponents
## NOTE: Somehow you need to transform size to relative size, without knowing opponent size.
            xs_rel = self.xs / self.estimate 
            for s in range(len(self.xs)):
                likelihood[s] = self._likelihood_function_wager(self.xs[s],outcome_params=fight.outcome_params)
        elif not win:
            #print('they lost!')
            e_self = fight.loser.effort
            w_opp = fight.winner.wager
            xs_rel = self.xs / max([fight.loser.size,fight.winner.wager / (self.effort**e)])
            for s in range(len(self.xs)):
                likelihood[s] = 1 - self._likelihood_function_wager(self.xs[s],outcome_params=fight.outcome_params)
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
    def _likelihood_function_se(self,x_size,o_size,x_eff = None,o_eff = None,fight=None):
## Check fight params
        if fight is None:
            s,e,l = self.naive_params
        else:
            s,e,l = fight.outcome_params
        if x_eff is None:
            x_eff = 1
        if o_eff is None:
            if fight.level is not None:
                o_eff = fight.level
            else:
                o_eff = 1
## Get relative sizes and wagers
        if x_size >= o_size:
            x_wag = x_eff**e
            o_wag = (o_size/x_size)**s * (o_eff**e)
        else:
            x_wag = (x_size/o_size)**s * (x_eff**e)
            o_wag = o_eff ** e
## Get relative wagers, and plug them into the wager_curve
        if x_wag > o_wag:
            rel_wag = o_wag / x_wag
            p_win = 1-self._wager_curve(rel_wag,l) ## because opponent is the underdog, 1-upset is x's p(win)
        elif x_wag == o_wag:
            return 0.5
        else: ## x_wager is smaller, so they're the underdog
            rel_wag = x_wag / o_wag
            p_win = self._wager_curve(rel_wag,l)
        if False and fight.p_win is not None:
            if fight.fish1.size == o_size:
                other_fish = fight.fish1
                me = fight.fish2
            elif fight.fish2.size == o_size:
                other_fish = fight.fish2
                me = fight.fish1
            else:
                print('\nummmm.....')
            print('\nTESTING STUFF:')
            import copy
            fight_ = copy.deepcopy(fight)
            if fight_.fish1.size == o_size:
                fight_.fish2.size = x_size
            else:
                fight_.fish1.size = x_size
            _ = fight_.mathy()


            print('my size,opp size,my effo, opp effort, my wager, opp wager')
            print(x_size,o_size,x_eff,o_eff,x_wag,o_wag)
            print('perceived p_win',p_win)
            print('actual p_win for smaller investor',fight_.p_win)
            print('actual efforts:',me.effort,other_fish.effort)
        return p_win

    def _use_mutual_likelihood(self,fight,win=True):
        if self.likelihood_dict is None:
            #likelihood = self._define_likelihood_mutual(fight,win) 
            likelihood = self._define_likelihood_mut_array(fight,win)
        elif True or (fight.winner.idx,fight.loser.idx) not in self.likelihood_dict.keys():
            #likelihood = self._define_likelihood_mutual(fight,win) 
            likelihood = self._define_likelihood_mut_array(fight,win)

        else: ## The dict is fast, but it doesn't work, I would need every possible effort in there too...
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
        if self.effort == 0:
            likelihood = np.ones(len(xs))
        if win:
            for s in range(len(xs)):
                if False:
                    likelihood[s] = self._likelihood_function_size(xs[s],x_opp)
                else:
                    likelihood[s] = self._likelihood_function_se(xs[s],x_opp,x_eff=self.effort,fight=fight)
        elif not win:
            for s in range(len(xs)):
                if False:
                    likelihood[s] = 1-self._likelihood_function_size(xs[s],x_opp)
                else:
                    likelihood[s] = 1- self._likelihood_function_se(xs[s],x_opp,o_eff=1,x_eff=self.effort,fight=fight)
        return likelihood
   
## Wager function optimized for array multiplication
    def _wager_curve_smart(self,w,L=np.tan((np.pi - .25)/2)):
        return (w ** L) / 2

## Updated likelihood function that *should* be faster
    def _define_likelihood_mut_array(self,fight,win=True):
        if fight is None:
            s,e,l = self.naive_params
        else:
            s,e,l = fight.outcome_params
        if l == 0:
            print('#### l == 0, this is a little weird....')
            return np.ones_like(self.xs)
        if self.effort == 0:
            return np.ones_like(self.xs)
        if self.effort == None:
            x_eff = 1
        else:
            x_eff = self.effort

## Define other fish size and effort based on fight
        if win:
            other_fish = fight.loser
            o_eff = fight.level 
        else:
            other_fish = fight.winner 
            o_eff = 1 ## this is an assumption of how much the other fish would have fought.
        if self.insight == True:
            o_eff = other_fish.effort
        xs = self.xs ## for simplicity, although sometimes I forget it exists
        likelihood = np.empty_like(xs)

## Build arrays of rel sizes of each fish as focal fish increases size
        size_index = np.argmax(self.xs >= other_fish.size)
        rel_xs = np.ones_like(xs)
        rel_os = np.ones_like(xs)
        rel_xs[:size_index] = self.xs[:size_index] / other_fish.size
        rel_os[size_index:] = other_fish.size / self.xs[size_index:] 
## Get wager arrays based on relative sizes
        x_wager = rel_xs ** s * x_eff ** e ## Wager of focal fish
        o_wager = rel_os ** s * o_eff ** e ## Wager of opponent

## Build relative wager array
        wager_array = np.empty_like(xs)
        if x_wager[-1] < o_wager[-1]:
            wager_index = len(o_wager) ## this deals with the case where x_wager is never bigger
            wager_array = x_wager / o_wager
        else:
            wager_index = np.argmax(x_wager > o_wager) ## the point x_wager becomes bigger
            wager_array[:wager_index] = x_wager[:wager_index] / o_wager[:wager_index]
            wager_array[wager_index:] = o_wager[wager_index:] / x_wager[wager_index:]

        L = np.tan((np.pi - l)/2) ## calculate this once to speed things up
        likelihood = self._wager_curve_smart(wager_array,L)
        if win: ## since likelihood is the probability of what happened, and wager_array was p(upset)
            likelihood[wager_index:] = 1 - likelihood[wager_index:]
        else:
            likelihood[:wager_index] = 1 - likelihood[:wager_index]
        return likelihood

## This also includes opponent assesment...need to fix that
    def update_prior_old(self,win,x_opp=False,xs=None,w_opp=None,e_self=None,outcome_params=None):
        if xs is None:
            xs = self.xs

## Get likelihood function (there are a couple possible methods)
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
   
    def update_energy(self,win,fight):
        if win:
            other_fish = fight.loser
            if self.size >= other_fish.size: 
                #cost = min([fight.fish1.wager,fight.fish2.wager])
                cost = min([self.effort,other_fish.wager]) 
            else: ## if a smaller fish wins, the bigger fish's effort is scaled up, but no more than effort spent
                #print('How did I lose???')
                #print(self.effort,other_fish.size,other_fish.effort,fight.params)
                cost = min([self.effort,other_fish.wager * (other_fish.wager / self.wager)])
            if fight.food is not None:
                if fight.food > 0:
                    self.energy = np.round(self.energy - cost + fight.food,2)
                    self.energy = np.clip(self.energy,0,self.max_energy)
                    self.size = self.size + self.r_rhp * (self.s_max - self.size) ** self.a_growth
                    self.fitness_record.append(0)
                else:
                    self.fitness_record.append(1)
        else:
            other_fish = fight.winner
            cost = self.effort
            if fight.food:
                self.energy = np.round(self.energy - self.effort,2)
        if self.energy_cost is False:
            self.energy = self.max_energy 
        if self.energy <= 0:
            #print('I am dying!',fight.level,self.effort)
            self.energy = 0
            self.alive = False
        self.size_record.append(self.size)
        self.energy_record.append(self.energy)

        self.win_record.append([other_fish.size,win,self.effort,cost])
        return other_fish

## For testing what happens if you just don't update ever
    def no_update(self,win,fight):

        if False and self.effort_method == 'Perfect':
            if win:
                print('I won!')
            else:
                print('I lost??')
                print(fight.fish1.size,fight.fish2.size)
                print(fight.fish1.effort,fight.fish2.effort)
                print(fight.fish1.wager,fight.fish2.wager)
                print('calculated p_win:',fight.p_win,'\n')
        self.update_energy(win,fight)
        return self.prior,self.estimate

# Updates the prior and handles food and cost. Should probably be rebuilt eventually
    def update_prior(self,win,fight):
        size_idx = np.argmax(self.xs >= self.size)
        size_possible_pre = self.prior[size_idx] > 0
## Establish fishes and impose costs and benefits
        other_fish = self.update_energy(win,fight) # This currently updates win record, not ideal
## Get likelihood function
        if self.effort_method[1] == 0:

            likelihood = self._use_simple_likelihood(fight,win)
            i_estimate = np.argmax(self.xs > self.estimate)
        else:
            likelihood = self._use_mutual_likelihood(fight,win)

            #likelihood = self._define_likelihood_mutual(fight,win)
        pre_prior = self.prior
        self.prior = self._update(self.prior,likelihood,self.xs) ## this just multiplies prior*likelihood
        if self.decay_all:
            print('decaying...')
            self.prior = self._decay_flat(self.prior)
        self.cdf_prior = self._get_cdf_prior(self.prior)
        pre_estimate = self.estimate
        estimate = self.xs[np.argmax(self.prior)]
        post_estimate = estimate
        #self.estimate_ = np.sum(self.prior * self.xs / np.sum(self.prior))
        prior_mean,prior_std = self.get_stats()
        self.estimate_ =  prior_mean
        self.est_record_.append(prior_mean)

        self.sdest_record.append(prior_std)

        est_5,est_25,est_75,est_95 = self._get_range_prior(self.cdf_prior) 
        self.range_record.append([est_5,est_25,est_75,est_95])
        
        self.estimate = estimate
        self.est_record.append(estimate)
        size_possible_post = self.prior[size_idx] > 0
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

    def prob_win_wager(self,my_size,opp_size,fight_params=None,opp_effort=0.5,own_effort=0.5):
        if fight_params is None:
            fight_params = self.naive_params
        s,e,l = fight_params
        bigger_size = max([my_size,opp_size])
        my_rel_size = my_size/bigger_size
        opp_rel_size = opp_size/bigger_size
## Note that we're conservative on effort, to avoid death vs hawks
        my_wager = (my_rel_size * s) * (own_effort * self.energy * e)
        opp_wager = (opp_rel_size * s) * (opp_effort * e)
        min_wager = min([my_wager,opp_wager])
        min_normed = min_wager / max([my_wager,opp_wager,.00001])
        if my_wager == opp_wager:
            p_upset = 0.5
        else:
            p_upset = self._wager_curve(min_normed,l)
        if my_wager == min_wager:
            p_win = p_upset
        else:
            p_win = 1 - p_upset 
        return p_win

## This function is a mess. Need to break it up for sure.
    def choose_effort(self,f_opp,strategy=None):
        if self.discrete:
            return self.choose_effort_discrete(f_opp,strategy)
        if strategy is None:
            strategy = self.effort_method
## The latter strategy here is more in keeping with the probabilistic mutual assessment, just against an average fish

        my_size = self.estimate
        if self.acuity == 0:
            opp_size = f_opp.size
        else:
            opp_size = np.clip(np.random.normal(f_opp.size,self.acuity),7,100)
        self.opp_estimate = opp_size
        if strategy == 'Perfect' or strategy == 'Estimate':
            #print('Choosing effort:')
            #print('my size:',self.size)
            #print('opp size:',f_opp.size)
            #s,e,l = self.naive_params
            if strategy == 'Estimate':
                ## Now you have to estimate with some error

                my_size = self.estimate ## You only have to do this once
                opp_size = np.random.normal(f_opp.size,self.awareness)
                opp_size = np.clip(opp_size,7,99)
                #print('guess vs self:',my_size,self.size)
                #print('guess vs opp:',opp_size,f_opp.size)
            else:
                my_size = self.size
                opp_size = f_opp.size
            p_win = self.prob_win_wager(my_size,opp_size,self.naive_params)
            effort = p_win
        elif strategy == [1,0]:
            #effort = self.estimate / 100
            effort = 1 - self.cdf_prior[np.argmax(self.xs > self.naive_estimate)]
        elif strategy == [0,1]:
            effort = 1 - opp_size / 100
        elif strategy == [1,1]:
#NOTE: I think cdf_prior is still a bit off, since it's summing to a very large number. 

## Currenty, this is the probability of being bigger. I need the probability of winning.
            effort = 1 - self.cdf_prior[np.argmax(self.xs > opp_size)]
            effort = np.round(effort,4)
            """
            if effort > .5 and self.estimate < f_opp.size:
                import pdb
                pdb.set_trace()
            """
            #effort =  np.sum(self.cdf_prior[self.xs > f_opp.size]) / np.sum(self.cdf_prior)

        elif strategy == 'BayesMA':
## Here, we iterate over the possible conditions. I expect this to be slow...
            #print('sum of prior:',np.sum(self.prior))
            p_win = 0
            sum_of_sizes = 0
            for x in range(len(self.xs)):
                #print(self.prior[x],self.prob_win_wager(x,opp_size,self.naive_params))
                p_x = self.prior[x]
                p_win_at_x = self.prob_win_wager(self.xs[x],opp_size,self.naive_params)

                sum_of_sizes += p_win_at_x
                p_win += p_x * p_win_at_x
                #p_win += self.prior[x] * self.prob_win_wager(x,opp_size,self.naive_params)
            #print('summed probability:',p_win)
            #print('mean prob_win_wager:',sum_of_sizes / len(self.xs))
            effort = p_win
            if False:
                print('Bayes p_win for',self.idx,self.estimate_,opp_size,p_win)
                fig,ax = plt.subplots()
                ax.plot(self.xs,self.prior)
                plt.show()
            #print('self_size,opp_size,opp_guess,effort (pre energy):',self.size,f_opp.size,opp_size,effort)
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
    
    def explore_effort(self,f_opp):
        if np.random.random() > .9:
            effort = np.random.random()
        else:
            effort = self.choose_effort(f_opp)

        return effort

    def choose_effort_energy(self,f_opp,strategy=None):
        effort = self.choose_effort(f_opp,strategy)
## A slightly more careful strategy, invests a proportion of available energy, should generally avoid death
        if False:
            effort = self.energy * (effort ** self.c_aversion)
            effort = np.clip(effort,0,self.energy)
        return effort

    def leroy_jenkins(self,f_opp):
        effort= self.energy
        return effort

    def half_jenkins(self,f_opp):
        effort = self.energy * .5
        return effort

    def float_jenkins(self,f_opp):
        effort = self.effort
        return effort

    def random_effort(self,f_opp):
        effort = np.random.random() * self.energy
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
    print(f1.idx,f1.size,len(f1.prior))
