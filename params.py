
## Load required packages

class Params():
    def __init__(self,n_iterations=1000,                    ## Sim Params
                n_fish=4,n_rounds=200,f_method='random',    ## Tank params
                energy_refill=0.5,energy_cost=True,
                f_outcome='math',outcome_params=[.3,.3,.3], ## Fight Params
                effort_method=[1,1],update_method='bayes',  ## Fish Params
                age=50,size=None,prior=None,likelihood=None,likelihood_dict=None,
                xs=np.linspace(7,100,500),r_rhp=0,a_growth=True,c_aversion=1,
                max_energy=1,start_energy=0.5,
                acuity=10,pre_acuity=10,post_acuity=1,awareness=10,insight=True,

                verbose=False):                             ## Other params
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
