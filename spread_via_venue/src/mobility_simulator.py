import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import math
import pandas as pd
import gc
from scipy import ndimage
import inspect

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['legend.numpoints'] = 1
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['font.size'] = 24
matplotlib.rcParams['font.family'] = 'sans serif'
matplotlib.rcParams['xtick.labelsize'] = 22
matplotlib.rcParams['ytick.labelsize'] = 22

## Constants
STATE_S = 100 # Suseptible
STATE_E = 101 # Exposed
STATE_I = 102 # Infected
STATE_C = 103 # Cared for / Isolated
STATE_R = 104 # Recovered
state_str = {STATE_S:'STATE_S', STATE_E:'STATE_E', STATE_I:'STATE_I', STATE_C:'STATE_C', STATE_R:'STATE_R'}

LOCKDOWN_0 = 0
LOCKDOWN_1 = 1 # Random subsample of checkins
LOCKDOWN_2 = 2 # Close popular venues
LOCKDOWN_3 = 3 # Isolate popular users
LOCKDOWN_4 = 4 # Cleaning
LOCKDOWN_5 = 5 # Different user groups operate in cycle
LOCKDOWN_6 = 6 # Random subsamples of venues are discarded
LOCKDOWN_7 = 7 # Random subseamples of users are discarded
LOCKDOWN_8 = 8 # Skip random checkins -- vary lockdown offset, length, and skip probability
LOCKDOWN_9 = 9 # Skip probabilities are given in the arg (from Google mobility report)
LOCKDOWN_12 = 12 # Popularity of a person is # of meetings

MAX_DAY = 500 # Maximum simulation days

class Configs:
    def __init__(self, num_seeds, infect_prob, reset_interval, \
                       asymp_prob, lockdown_num, lockdown_arg, lockdown_offset, lockdown_len, DBG=False):
        self._num_seeds = num_seeds
        self._infect_prob = infect_prob
        self._reset_interval = reset_interval
        self._asymp_prob = asymp_prob
        self._lockdown_num = lockdown_num
        self._lockdown_arg = lockdown_arg
        self._lockdown_offset = lockdown_offset
        self._lockdown_len = lockdown_len
        self._city = None
        self._DBG = DBG
    
    def print(self, *argv):
        if (self._DBG == True):
            print ("["+inspect.stack()[1][3] + "] ", *argv)

    
    def __str__(self):
        return "_num_seeds:"+str(self._num_seeds)+"\n" +\
               "_reset_interval:" + str(self._reset_interval)+"\n"+\
               "_infect_prob:"+str(self._infect_prob)+"\n"+\
               "_asymp_prob:"+str(self._asymp_prob)+"\n"+\
               "_lockdown_num, arg:("+ str(self._lockdown_num) + ","+ str(self._lockdown_arg) +")\n"+\
               "_lockdown_offset, len:("+str(self._lockdown_offset)+", "+str(self._lockdown_len)+")\n"


class User:
    next_state_table = {STATE_S:STATE_E, STATE_E:STATE_I, STATE_I:[STATE_C, STATE_R], STATE_C:STATE_R, STATE_R:None}
    
    def __init__(self, config):
        self.state = STATE_S
        self.config = config
        self.group = None # This is only related when grouping
        
        if (np.random.uniform() <= self.config._asymp_prob):
            self.is_symp = False # True: symptomatic, False: asymptomatic
        else:
            self.is_symp = True

        self.infect_from_venue = None
        self.expose_time = 0  # when the user is exposed
        self.infectious_time = 0 # after incubation period
        self.incare_time = 0 # when moves to care / self-isolate
        self.recover_time = 0 # when she is recovered
        
        self.infected_from_me = 0
    
    def _validate_state(self):
        if (self.state not in [STATE_S, STATE_E, STATE_I, STATE_C, STATE_R]):
            print ("Unknown state:", self.state)
            raise Exception

    def sample_time(self, mean, std, left_bound):
        sample = np.random.normal(loc = mean, scale=std)
        if (sample < left_bound):
            sample = left_bound
        return sample
    
    def _implicit_state_change(self, curtime):
        if (self.is_symp == True):
            if (self.state == STATE_E and self.infectious_time > 0  and curtime >= self.infectious_time):
                self.state = STATE_I
            if (self.state == STATE_I and self.incare_time > 0 and curtime >= self.incare_time):
                self.state = STATE_C
            if (self.state == STATE_C and self.recover_time > 0 and curtime >= self.recover_time):
                self.state = STATE_R
        else:
            if (self.state == STATE_E and self.infectious_time > 0  and curtime >= self.infectious_time):
                self.state = STATE_I
            if (self.state == STATE_I and self.recover_time > 0 and curtime >= self.recover_time):
                self.state = STATE_R

    def expose(self, curtime, is_seed = False):
        if (self.state != STATE_S):
            raise Exception("Infecting with state:"+str(self.state))

        self.expose_time = curtime
        if (is_seed == False):
            self.state = STATE_E
            self.infectious_time = self.sample_time(curtime+6*24, 1*24, curtime)
        else:
            self.state = STATE_I
            self.infectious_time = curtime

        if (self.is_symp == True):
            #self.incare_time = self.sample_time(self.infectious_time + 5*24, 0.5*24, self.infectious_time)
            #self.recover_time = self.sample_time(self.incare_time + 13*24, 0.5*24, self.incare_time)
            
            self.incare_time = self.sample_time(curtime + 11*24, 1*24, self.infectious_time)
            self.recover_time = self.sample_time(curtime + 24*24, 1*24, self.incare_time)
        else:
            #self.recover_time = self.sample_time(self.infectious_time + 17*24, 0.5*24, self.infectious_time)
            
            self.recover_time = self.sample_time(curtime + 24*24, 1*24, self.infectious_time)
        
        self._implicit_state_change(curtime)


    def get_state(self, curtime):
        self._implicit_state_change(curtime)
        return self.state


    def __str__(self):
        # STATE_E-> infected STATE_I-> infectious
        return state_str[self.state]+ " is_symp:"+str(self.is_symp)+" expose_time:"+str(self.expose_time)+\
               " infectious_time:"+str(self.infectious_time)+" incare_time:"+str(self.incare_time)+\
               " recover_time:"+str(self.recover_time)+ "infected_from_me:"+str(self.infected_from_me)+"\n"
        

    def __repr__(self):
        return self.__str__()


class Venue:
    def __init__(self, config):
        self.config = config;
        self.last_infect_time = 0;
        self.last_infected_user = None;
        self.visited_users = [];
        if (self.config._lockdown_num == LOCKDOWN_4):
            self.last_clean_time = np.random.uniform(0, self.config._lockdown_arg);

    def update_last_clean_time(self, curtime):
        if (self.config._lockdown_num == LOCKDOWN_4):
            self.last_clean_time += int(math.floor((curtime - self.last_clean_time)\
                                    /self.config._lockdown_arg))*self.config._lockdown_arg

    def reset_last_infect_time(self, curtime):
        if (self.last_infect_time > 0 and curtime - self.last_infect_time > self.config._reset_interval):
            #print ("reset_last_infect_time: diff:", curtime - self.last_infect_time, "config._reset_interval:", config._reset_interval)
            self.last_infect_time = 0
            self.last_infected_user = None

    def get_last_infect_time(self, curtime):
        if (self.config._lockdown_num == LOCKDOWN_4):
            self.update_last_clean_time(curtime)
            if (self.last_clean_time > self.last_infect_time):
                self.last_infect_time = 0 # Reset timer
                self.last_infected_user = None
        self.reset_last_infect_time(curtime)
        return self.last_infect_time, self.last_infected_user

    def infect_venue(self, curtime, userid):
        self.last_infected_user = userid
        self.last_infect_time = curtime

    def __str__(self):
        return ' last_infect_time:' + str(self.last_infect_time)+ " last_clean_time:"\
                +str(self.last_clean_time) + " last_infected_user:"+str(self.last_infected_user)+"\n"
    
    def __repr__(self):
        return self.__str__()

class MobilitySimulator:
    def __init__(self, df, start_time, nusers, nvenues, ncheckins, config):
        self.config = config
        self.user_dict = {}
        self.venue_dict = {}
        #self.filename = filename
        self.df = df.sort_values('posixtime', ascending=True).reset_index(drop=True)
        #print (df)
        #exit(0)
        self.start_time = start_time
        self.nusers = nusers
        self.nvenues = nvenues
        self.ncheckins = ncheckins
        self.lock_down_start_day = -1
        '''
        #self.df = df.sort_values('posixtime', ascending=True).reset_index(drop=True)
        self.start_time = self.df.iloc[0]['posixtime']

        # FIXME: Change these to given fixed parameters
        self.num_venues = df.groupby("venueId").ngroups
        self.num_users = df.groupby("userId").ngroups
        '''
        
        #self.seeds = df.head(self.config._num_seeds)['userId'].tolist()

    def create_new_user(self, userid):
        self.user_dict[userid] = User(self.config)
        #if (self.config._lockdown_num == LOCKDOWN_5):
        #    self.user_dict[userid].group = np.random.choice(self.config._lockdown_arg)
 
    def create_new_venue(self, venueid):
        self.venue_dict[str(venueid)] = Venue(self.config)
 
    def is_within_lockdown(self, today):
        if (self.config._lockdown_len == 0):
            return True # Length is not set. The whole duration is lockdown.
        if (today >= self.lock_down_start_day and \
            today <= self.lock_down_start_day + self.config._lockdown_len):
            return True
        return False
            
    
    def get_blacklist_venues(self):
        if (self.config._lockdown_num in [LOCKDOWN_2]):
            if (self.config._lockdown_num == LOCKDOWN_2):
                rank = int(math.ceil(self.nvenues*self.config._lockdown_arg/100.))
            
            xf = self.df.groupby('venueId').count()[['userId']]
            sorted_xf = xf.sort_values('userId', ascending=False)
            blacklist_venues = sorted_xf.index.tolist()[:rank]
            return blacklist_venues
        elif (self.config._lockdown_num == LOCKDOWN_6):
            all_venues = self.df.venueId.unique()
            num_venues = int(len(all_venues)*self.config._lockdown_arg/100.)
            blacklist_venues = np.random.choice(all_venues, replace=False, size=num_venues)
            
            return blacklist_venues
            

    def find_most_meeting_popular(self, rank):
        meeting_df = pd.read_csv('../data/'+self.config._city+'/meeting_count.csv')
        return meeting_df[:rank].userId.to_list()

    def find_most_checkin_popular(self, rank):
        #df = pd.read_csv(PROJ_HOME+"data/"+city+"/checkin.csv")
        xf = self.df.groupby('userId').count()[['venueId']]
        sxf = xf.sort_values('venueId', ascending=False)
        sxf.rename(columns={'venueId': 'popularity'}, inplace=True)
        sxf.index.names = ['userId']
        return sxf[:rank].index.to_list()

    
    
    def get_isolated_users(self):
        percentage = self.config._lockdown_arg
        
        num_to_select = int(math.ceil(self.nusers*percentage/100.))
        
        if (self.config._lockdown_num in [LOCKDOWN_3]):
            return self.find_most_checkin_popular(num_to_select)

        elif (self.config._lockdown_num == LOCKDOWN_7):
            all_users = self.df.userId.unique()
            blacklist_users = np.random.choice(all_users, replace=False, size=num_to_select)
            return blacklist_users

        elif (self.config._lockdown_num == LOCKDOWN_12):
            #meeting_df = pd.read_csv("../data/"+self._city+"/meeting_dict.csv")
            #sorted_xf = meeting_df.sort_values('meeting_count', ascending=False)
            return self.find_most_meeting_popular(num_to_select)

    # While selecting seeds consider not to choose the blacklisted users
    def select_seeds(self, isolated_users):
        #users = set(self.df[self.df['posixtime'] <= 2*24].userId.unique())
        users = set(self.df.userId.unique())
        if (len(isolated_users) > 0):
            users = users.difference(set(isolated_users))
        users = list(users)
        return np.unique(np.random.choice(users, size=self.config._num_seeds, replace=False)).astype('int').tolist()
        

    def select_seed_indices(self):
        #print("self.config._num_seeds:", self.config._num_seeds)
        return np.random.choice(self.config._num_seeds*10, size=self.config._num_seeds, replace=False)
    
    def run(self):
        infection_count = 0
        valid_checkin_counts_with_incare = np.zeros(MAX_DAY)
        
        blacklist_venues, isolated_users, blacklist_checkins = [], [], []

        if (self.config._lockdown_num in [LOCKDOWN_2, LOCKDOWN_6]):
            blacklist_venues = self.get_blacklist_venues()
        if ( self.config._lockdown_num in [LOCKDOWN_3, LOCKDOWN_7, LOCKDOWN_12]):
            isolated_users = self.get_isolated_users()

        print ("len(blacklist_venues):", len(blacklist_venues), \
                            "len(isolated_users):", len(isolated_users), \
                            "len(blacklist_checkins):", len(blacklist_checkins))
        seeds = self.select_seeds(isolated_users)
        #print ("seeds:", seeds)
        using_seed_indices = False
        seed_indices = self.select_seed_indices()
        #self.config.print (seed_indices)
        yesterday = -1
        yesterday_infect_count = 0
        
        #chunksize = 10 ** 6
        #for df in pd.read_csv(self.filename, chunksize=chunksize, dtype={'userId': np.int32, 'venueId':np.str, 'posixtime':np.float64}):
        for index, row in self.df.iterrows():
            userid = row['userId']
            venueid = row['venueId']
            checkin_time = row['posixtime']

            if (self.user_dict.get(userid, None) == None):
                self.create_new_user(userid)
            if (self.venue_dict.get(venueid, None) == None):
                self.create_new_venue(venueid)

            #self.config.print(checkin_time, venueid, userid, str(self.user_dict[userid])[:-1])
            
            today = int(math.floor((checkin_time - self.start_time)/24.0))
            
            if (today != yesterday):
                if (today % 50 == 0):
                    print ( today, "infection_count:", infection_count,\
                        "index:", index, "valid checkins:", np.sum(valid_checkin_counts_with_incare))
                yesterday = today

            if ( 100*infection_count/float(self.nusers) >= self.config._lockdown_offset):
                if (self.lock_down_start_day == -1):
                    self.lock_down_start_day = today
                    print ("lockdown started.", self.lock_down_start_day)

                if (self.is_within_lockdown(today) == True):
                    #print (today, "within lockdown.")

                    #if (self.config._lockdown_num == LOCKDOWN_1 and (index in blacklist_checkins)):
                    #    continue
                    if (venueid in blacklist_venues): # Lockdown 2
                        continue;
                    if (userid in isolated_users): # Lockdown 3, 10, 12
                        continue;
                    if (self.config._lockdown_num in [LOCKDOWN_1, LOCKDOWN_8]):
                        if (np.random.uniform() <= self.config._lockdown_arg): # Skip every check-in with prob.
                            continue;

                    #if (self.config._lockdown_num == LOCKDOWN_5):
                    #    if (today % self.config._lockdown_arg != self.user_dict[userid].group):
                    #        continue;
                    if (self.config._lockdown_num == LOCKDOWN_9):
                        if (today >= len(self.config._lockdown_arg)):
                            print ("BREAKING taday:", today)
                            break;
                        #print (today, self.config._lockdown_arg[today])
                        if (np.random.uniform() <= self.config._lockdown_arg[today]):
                            continue
            if (using_seed_indices == True):
                if (index in seed_indices):
                    if (self.user_dict[userid].get_state(checkin_time) == STATE_S):
                        self.user_dict[userid].expose(checkin_time, is_seed = True)

                        infection_count += 1
                        #self.config.print ("\t\ttoday:", today,", seed userid:", userid, "user:", str(self.user_dict[userid])[:-1])
            else:
                if (userid in seeds):
                    seeds.remove(userid)
                    if (self.user_dict[userid].get_state(checkin_time) == STATE_S):
                        self.user_dict[userid].expose(checkin_time, is_seed = True)
                        self.user_dict[userid].infectious_time = checkin_time
                        infection_count += 1
                        #print ("seeding by userid.")


            user_state = self.user_dict[userid].get_state(checkin_time)

            #if (user_state != STATE_C): # The user does not checkin when in care.
            valid_checkin_counts_with_incare[today] += 1
            
            if (user_state != STATE_S):
                if (user_state == STATE_I):
                    self.venue_dict[venueid].infect_venue(checkin_time, userid)
                    self.config.print ("\t\ttoday:", today, "**Infect the venue.")
                self.config.print ("\t\t", today, "**Do not proceed. user:", str(self.user_dict[userid])[:-1])
                continue
            
            last_infect_time, last_infected_user = self.venue_dict[venueid].get_last_infect_time(checkin_time)

            if (last_infect_time > 0):
                self.config.print ("\t\tvenueid:", venueid, self.venue_dict[venueid], "timediff:", (checkin_time - last_infect_time))
            
            if (last_infect_time > 0 and (checkin_time - last_infect_time) <= self.config._reset_interval):
                if (np.random.uniform() <= self.config._infect_prob):
                    self.user_dict[userid].expose(checkin_time)
                    self.user_dict[userid].infect_from_venue = venueid
                    self.user_dict[last_infected_user].infected_from_me += 1
                    infection_count += 1

                        
            if (today >= 150):
                break;
            
        print ("done. infection_count:", infection_count, "index:", index, "valid checkins:", np.sum(valid_checkin_counts_with_incare))
        return [self.user_dict, None, valid_checkin_counts_with_incare]



######################### Plot functions ######################################
class LabelConfig:
    def __init__(self, config_index, prefix="", suffix="", xlabel="", ylabel=""):
        self.prefix = prefix
        self.suffix = suffix
        self.config_index = config_index
        self.xlabel = xlabel
        self.ylabel = ylabel

class Plot:
    def __init__(self, result_dir, configs=None, label_config=None, start_time=0, nusers=0):
        self.result_dir = result_dir
        self.label_config = label_config
        self.start_time = start_time
        self.nusers = nusers
        self.configs = configs
    
    def get_label(self, _config):
        if (self.label_config.config_index == -1):
            return self.label_config.prefix+self.label_config.suffix
        if (_config[4] == 0 and _config[self.label_config.config_index] == 0):
            return "No mitigation"
        return self.label_config.prefix+str(_config[self.label_config.config_index])+self.label_config.suffix
    
    def set_lims(ax, xmin, xmax, ymin, ymax):
        _xmin, _xmax = ax.get_xlim()
        _ymin, _ymax = ax.get_ylim()
        if (xmin != None):
            _xmin = xmin
        if (xmax != None):
            _xmax = xmax
        if (ymin != None):
            _ymin = ymin
        if (ymax != None):
            _ymax = ymax
        ax.set_xlim([_xmin, _xmax])
        ax.set_ylim([_ymin, _ymax])
    
    ##### Static utility functions #####
    def daily_count(start_time, times, interval_hr=24.):
        count = np.zeros(MAX_DAY)
        for t in times:
            try:
                day = int((t-start_time)/interval_hr)
                count[day] += 1
            except Exception:
                print ("start_time:", start_time, "t:", t)
                exit(0)
        return count
    
    def get_active_cases(daily_infect, daily_remove):
        active_cases = np.zeros(len(daily_infect))
        active_cases[0] = daily_infect[0]
        for d in range(1, len(daily_infect)):
            active_cases[d] = active_cases[d-1] + daily_infect[d] - daily_remove[d-1]
        return active_cases
    
    def get_expose_recover_times(user_dict, exclude_seed=False):
        expose_times, recover_times = np.array([]), np.array([])
        for uid in user_dict.keys():
            if (user_dict[uid].expose_time > 0):
                expose_times = np.append(expose_times, user_dict[uid].expose_time)
                recover_times = np.append(recover_times, user_dict[uid].recover_time)
        return np.sort(expose_times), np.sort(recover_times)
    
    def get_doubling_time(cum_infects):
        doubling_times = []
        nd = len(cum_infects)
        for d in range(nd):
            x = cum_infects[d]
            # find the first day with infects 2*x
            for d1 in range(d, nd):
                if (cum_infects[d1] >= 2*x):
                    #print ("d:", d, "x:", x, "d1:", d1, "cum_infects[d1]:", cum_infects[d1], "doubling time:", d1-d)
                    break
            doubling_times.append(d1-d)
            if (d1 >= nd-1):
                break
        return np.asarray(doubling_times)
    '''
    def get_growth_factor(x):
        #N = 3
        #daily_infects = np.convolve(x, np.ones((N,))/N, mode='valid')
        daily_infects = x
        growth_factors = [daily_infects[0]]
        for d in range(1, len(daily_infects)):
            #print (d, daily_infects[d-1])
            if (daily_infects[d-1] > 0):
                gr = (daily_infects[d] - daily_infects[d-1])/float(daily_infects[d-1])
                #growth_factors.append( daily_infects[d] / float(daily_infects[d-1]) )
                growth_factors.append( gr )
            else:
                pass
                #growth_factors.append((daily_infects[d] - daily_infects[d-1]))
        growth_factors[0] = 0
        return growth_factors
    '''
    def get_rolling_avg(x, N):
        rolling_avg = []
        for d in range(len(x)-N):
            rolling_avg.append( sum(x[d:d+N]) / N )
        return np.asarray(rolling_avg)
    
    def get_growth_factor(x, N):
        rolling_avg = Plot.get_rolling_avg(x, N)
        
        growth_factors = [rolling_avg[0]]
        for d in range(1, len(rolling_avg)):
            if (rolling_avg[d-1] > 0 and rolling_avg[d] > 0):
                gr = (rolling_avg[d])/float(rolling_avg[d-1])
                #if (d < 10):
                #    print (gr, rolling_avg[d], float(rolling_avg[d-1]))
                growth_factors.append( gr )
            else:
                pass
        #print ("--------------------------------")
        #growth_factors[0] = 0
        return growth_factors[1:]
        
    def saveplotfile(figure, filename, transparent=False, lgd=None, bbox='tight', extension="pdf"):
        print ("saving:", filename)
        if (extension == "pdf"):
            if (lgd!=None):
                figure.savefig(filename+'.pdf', format='pdf', dpi=1000, transparent=True, bbox_extra_artists=(lgd,), bbox_inches=bbox)
            else:
                figure.savefig(filename+'.pdf', format='pdf', dpi=1000, transparent=True, bbox_inches=bbox)
        elif (extension == "png"):
            if (lgd!=None):
                figure.savefig(filename+'.png', format='png', dpi=600, transparent=transparent, bbox_extra_artists=(lgd,), bbox_inches=bbox)
            else:
                figure.savefig(filename+'.png', format='png', dpi=600, transparent=transparent, bbox_inches=bbox)

    
    def plot_legends(ax, ncol, outfilename):
        import matplotlib.pylab as pylab
        legend_fig = pylab.figure()
        legend = pylab.figlegend(*ax.get_legend_handles_labels(), ncol=ncol, loc='center')
        #legend.get_frame().set_color('0.70')
        legend_fig.canvas.draw()
        bbox  = legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted())
        if (outfilename != ""):
            Plot.saveplotfile(legend_fig, outfilename, bbox=bbox)
        ax.legend().set_visible(False)
        plt.draw()
    
    def cdf_exposed(self, nolockdown_config, N, ax=None, outfilename="", xmin=None, xmax=None, ymin=None, ymax = None, interval_hr=24., legend=False):
        if (ax == None):
            fig, ax = plt.subplots(figsize=(6,5))
        
        ax.locator_params(nbins=4)
        allconfigs = []
        if (len(nolockdown_config) > 0):
            allconfigs = [nolockdown_config]
        
        [allconfigs.append(_c) for _c in self.configs]
        final_ticks = []
        for _config in allconfigs:
            filename = "_".join([str(x) for x in _config])
            results = np.load(self.result_dir+filename+".npy", allow_pickle=True)
            cum_exposed = np.array([])
            for res in results:
                if (res[0] == None):
                    xx, _ = res[1]
                    daily_expose = Plot.get_rolling_avg(xx, N)*100/float(self.nusers)
                else:
                    expose_times, recover_times = Plot.get_expose_recover_times(res[0])
                    daily_expose = Plot.get_rolling_avg(Plot.daily_count(self.start_time, expose_times, interval_hr), N)*100/float(self.nusers)
                if (len(cum_exposed) == 0):
                    cum_exposed = np.cumsum(daily_expose)
                else:
                    cum_exposed = np.c_[cum_exposed, np.cumsum(daily_expose) ]
            #print ("cum_infects:", cum_infects.shape)
            mid = np.median(cum_exposed, axis=1)
            upper = np.percentile(cum_exposed, q=75, axis=1)
            lower = np.percentile(cum_exposed, q=25, axis=1)
            ax.plot(range(len(mid)), mid, label=self.get_label(_config))
            ax.fill_between(range(len(mid)), upper, lower, alpha=0.3)
            
            if (xmax == None):
                final_ticks.append(round(mid[-1]))
            else:
                final_ticks.append(round(mid[xmax]))
            #print ("-----------")
        ax2 = ax.twinx()
        Plot.set_lims(ax, xmin, xmax, ymin, ymax)
        ymin, ymax = ax.get_ylim()
        ax2.set_ylim([ymin, ymax])

        final_ticks = sorted(final_ticks)
        to_be_deleted_index = [] 
        for index in range(1, len(final_ticks)): 
             #print (final_ticks[index], final_ticks[index-1]) 
             if (final_ticks[index] - final_ticks[index - 1] < 4): 
                 to_be_deleted_index.append(index) 
        for ix in sorted(to_be_deleted_index, reverse=True): 
            del final_ticks[ix] 

        ax2.set_yticks(final_ticks)
        
        
        #ax.set_xlabel("Days from start")
        #ax.set_ylabel("% of people infected in total")
    
        if (legend == True):
            ax.legend()
        
        #Plot.plot_legends(ax, ncol, outfilename=outfilename+"_legends")
            
        #print ("filename:", filename)
        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)
        return ax

    def active_cases(self, nolockdown_config, N, ax=None, outfilename="", xmin=None, xmax=None, ymin=None, ymax = None, legend=False):
        if (ax == None):
            fig, ax = plt.subplots(figsize=(6,5))
        ax.locator_params(nbins=4)
        allconfigs = []
        if (len(nolockdown_config) > 0):
            allconfigs = [nolockdown_config]
        [allconfigs.append(_c) for _c in self.configs]

        print (allconfigs)
        for _config in allconfigs:#self.configs:
            filename = "_".join([str(x) for x in _config])
            results = np.load(self.result_dir+filename+".npy", allow_pickle=True)
            active_cases = np.array([])
            for res in results:
                if (res[0] == None):
                    xx, yy = res[1]
                    daily_expose = Plot.get_rolling_avg(xx, N)*100/float(self.nusers)
                    daily_recover = Plot.get_rolling_avg(yy, N)*100/float(self.nusers)
                else:
                    expose_times, recover_times = Plot.get_expose_recover_times(res[0])
                    daily_expose = Plot.get_rolling_avg(Plot.daily_count(self.start_time, expose_times), N)*100/float(self.nusers)
                    daily_recover = Plot.get_rolling_avg(Plot.daily_count(self.start_time, recover_times), N)*100/float(self.nusers)
                temp = Plot.get_active_cases(daily_expose, daily_recover)
                if (len(active_cases) == 0):
                    active_cases = temp
                else:
                    active_cases = np.c_[active_cases, temp ]
            #print ("active_cases:", active_cases.shape)
            mid = np.median(active_cases, axis=1)
            upper = np.percentile(active_cases, q=75, axis=1)
            lower = np.percentile(active_cases, q=25, axis=1)
            ax.plot(range(len(mid)), mid, label=self.get_label(_config))
            ax.fill_between(range(len(mid)), upper, lower, alpha=0.3)
            print("peak day:", np.argmax(mid))
            
            #print ("-----------")
        Plot.set_lims(ax, xmin, xmax, ymin, ymax)
        #ax.set_ylabel("Active cases (% population)", fontsize=21)
        #ax.set_xlabel("Days from start")
        if (legend == True):
            ax.legend()
        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)
        return ax


    def new_cases(self, nolockdown_config, N, ax=None, outfilename="", xmin=None, xmax=None, ymin=None, ymax = None, interval_hr=24., legend=False):
        if (ax == None):
            fig, ax = plt.subplots(figsize=(6,5))
        ax.locator_params(nbins=4)
        allconfigs = []
        if (len(nolockdown_config) > 0):
            allconfigs = [nolockdown_config]
        
        [allconfigs.append(_c) for _c in self.configs]
        final_ticks = []
        for _config in allconfigs:
            filename = "_".join([str(x) for x in _config])
            results = np.load(self.result_dir+filename+".npy", allow_pickle=True)
            new_exposed = np.array([])
            for res in results:
                if (res[0] == None):
                    xx, _ = res[1]
                    daily_expose = Plot.get_rolling_avg(xx, N)*100/float(self.nusers)
                else:
                    expose_times, recover_times = Plot.get_expose_recover_times(res[0])
                    daily_expose = Plot.get_rolling_avg(Plot.daily_count(self.start_time, expose_times, interval_hr), N)*100/float(self.nusers)
                
                #daily_expose = ndimage.gaussian_filter1d(daily_expose, 2)
                if (len(new_exposed) == 0):
                    new_exposed = daily_expose
                else:
                    new_exposed = np.c_[new_exposed, daily_expose ]
            #print ("cum_infects:", cum_infects.shape)
            mid = np.median(new_exposed, axis=1)
            upper = np.percentile(new_exposed, q=75, axis=1)
            lower = np.percentile(new_exposed, q=25, axis=1)
            ax.plot(range(len(mid)), mid, label=self.get_label(_config))
            ax.fill_between(range(len(mid)), upper, lower, alpha=0.3)
            
            if (xmax == None):
                final_ticks.append(round(mid[-1]))
            else:
                final_ticks.append(round(mid[xmax]))
            #print ("-----------")
        
        Plot.set_lims(ax, xmin, xmax, ymin, ymax)
        ymin, ymax = ax.get_ylim()

        #ax.set_xlabel("Days from start")
        #ax.set_ylabel("New cases (% population)", fontsize=22)
    
        if (legend == True):
            ax.legend()
        
        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)
        return ax



    def plot_growth_factor(self, nolockdown_config, N, ax=None, outfilename="", xmin=None, xmax=None, ymin=None, ymax = None):
        if (ax == None):
            fig, ax = plt.subplots(figsize=(6,5))
        ax.locator_params(nbins=4)
        allconfigs = []
        if (len(nolockdown_config) > 0):
            allconfigs = [nolockdown_config]
        ax.axhline(y=1, color='k', ls='--')
        
        [allconfigs.append(_c) for _c in self.configs]
        val_max = []; val_min = []
        for _config in allconfigs:
            filename = "_".join([str(x) for x in _config])
            results = np.load(self.result_dir+filename+".npy", allow_pickle=True)
            growth_factor = np.array([])
            for res in results:
                if (res[0] == None):
                    xx, _ = res[1]
                    daily_expose = Plot.get_rolling_avg(xx, N)*100/float(self.nusers)
                else:
                    expose_times, recover_times = Plot.get_expose_recover_times(res[0], exclude_seed=True)
                    daily_expose = Plot.daily_count(self.start_time, expose_times)
                    #daily_recover = Plot.daily_count(self.start_time, recover_times)
                temp = Plot.get_growth_factor(daily_expose, N)
                pad = np.zeros(MAX_DAY - len(temp))
                temp = np.append(temp, pad)
                temp = ndimage.gaussian_filter1d(temp, 2)
                if (len(growth_factor) == 0):
                    growth_factor = temp
                else:
                    growth_factor = np.c_[growth_factor, temp ]
            
            mid = np.median(growth_factor, axis=1)
            val_max.append(max(mid[xmin:xmax])); val_min.append(min(mid[xmin:xmax]))
            upper = np.percentile(growth_factor, q=75, axis=1)
            lower = np.percentile(growth_factor, q=25, axis=1)
            ax.plot(range(len(mid)), mid, label=self.get_label(_config))
            ax.fill_between(range(len(mid)), upper, lower, alpha=0.3)
            #print ("-----------")
        
        ymax = 0.1+max(val_max); ymin = min(val_min)-0.1
        print ("ymax:", ymax, "ymin:", ymin)
        Plot.set_lims(ax, xmin, xmax, ymin, ymax)
        
        #ax.set_ylabel("Growth rate ($\lambda_t$)")
        #ax.set_xlabel("Days from start")

        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)
        return ax

    def social_value(self, df, nolockdown_config, ax=None, outfilename="",\
                     xmin=None, xmax=None, ymin=None, ymax = None, interval_hr=24., legend=False):
        if (ax == None):
            fig, ax = plt.subplots() #figsize=(6,5))
        ax.locator_params(nbins=4)
        all_checkins = np.zeros(MAX_DAY)
        for ct in df.posixtime.tolist():
            _d = int((ct-self.start_time)/24.)
            all_checkins[_d] += 1
        
        allconfigs = [nolockdown_config]
        [allconfigs.append(_c) for _c in self.configs]

        
        zero_days = np.where(all_checkins == 0)[0]
        xs = np.arange(MAX_DAY); xs = np.delete(xs, zero_days)
        
        for _config in allconfigs:
            filename = "_".join([str(x) for x in _config])
            results = np.load(self.result_dir+filename+".npy", allow_pickle=True)
            counts_with = np.array([])

            for _, _, x2 in results:
                
                x2 = 100*np.divide(x2, all_checkins, out=np.zeros_like(x2), where=all_checkins!=0)
                x2 = np.delete(x2, zero_days)
                
                x2 = ndimage.gaussian_filter1d(x2, 2)
                if (len(counts_with) == 0):
                    counts_with = x2
                else:
                    counts_with = np.c_[ counts_with, x2 ]

            mid2 = np.median(counts_with, axis=1)
            upper2 = np.percentile(counts_with, q=75, axis=1)
            lower2 = np.percentile(counts_with, q=25, axis=1)
            ax.plot(range(len(mid2)), mid2, label=self.get_label(_config))
            ax.fill_between(range(len(mid2)), upper2, lower2, alpha=0.3)
                        
            #print ("-----------")
        Plot.set_lims(ax, xmin, xmax, ymin, ymax)
        #ax.set_xlabel("Days from start")
        #ax.set_ylabel("% check-ins remain")
        if (legend == True):
            ax.legend()
        #print ("filename:", filename)
        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)
        return ax


    '''
    def Rt(self, nolockdown_config, ax=None, outfilename="", xmin=None, xmax=120, ymin=None, ymax = None, legend=False):
        if (ax == None):
            fig, ax = plt.subplots()
        
        allconfigs = []
        if (len(nolockdown_config) > 0):
            allconfigs = [nolockdown_config]
        
        [allconfigs.append(_c) for _c in self.configs]
        
        for _config in allconfigs:
            filename = "_".join([str(x) for x in _config])
            results = np.load(self.result_dir+filename+".npy", allow_pickle=True)
            Rts = np.array([])
            for user_dict, _, _ in results:
                _infects_from_me = np.zeros(MAX_DAY)
                _infected_today = np.zeros(MAX_DAY)
                for uid in user_dict.keys():
                    if (user_dict[uid].expose_time > 0):
                        day = int((user_dict[uid].expose_time - self.start_time)/24.)
                        _infects_from_me[day] += user_dict[uid].infected_from_me
                        _infected_today[day] += 1
                Rt = np.array([])
                for d in range(MAX_DAY):
                    if (_infected_today[d] > 0):
                        Rt = np.append(Rt, _infects_from_me[d] / _infected_today[d])
                    else:
                        Rt = np.append(Rt, 0)
                Rt = ndimage.gaussian_filter1d(Rt, 1)
                #Rt = ndimage.uniform_filter1d(Rt, 4)
                #print (Rt)
                #exit(0)
                
                if (len(Rts) == 0):
                    Rts = Rt
                else:
                    Rts = np.c_[Rts, Rt]
            #print ("active_cases:", active_cases.shape)
            mid = np.median(Rts, axis=1)
            upper = np.percentile(Rts, q=75, axis=1)
            lower = np.percentile(Rts, q=25, axis=1)
            ax.plot(range(len(mid)), mid, label=self.get_label(_config))
            ax.fill_between(range(len(mid)), upper, lower, alpha=0.3)
            #print ("-----------")
        Plot.set_lims(ax, xmin, xmax, ymin, ymax)
        ax.set_ylabel("$R_t$")
        ax.set_xlabel("Days from start")
        if (legend == True):
            ax.legend()
        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)
        return ax
    '''



    '''
    def plot_doubling_times(self, nolockdown_config, ax=None, outfilename="", xmin=None, xmax=None, ymin=None, ymax = None, legend=False):
        if (ax == None):
            fig, ax = plt.subplots()

        allconfigs = []
        if (len(nolockdown_config) > 0):
            allconfigs = [nolockdown_config]
        
        [allconfigs.append(_c) for _c in self.configs]
        
        for _config in allconfigs:
            filename = "_".join([str(x) for x in _config])
            results = np.load(self.result_dir+filename+".npy", allow_pickle=True)
            doubling_times = np.array([])
            for res in results:
                expose_times, recover_times = Plot.get_expose_recover_times(res[0])
                daily_expose = Plot.daily_count(self.start_time, expose_times)*100/float(self.nusers)
                temp = Plot.get_doubling_time(np.cumsum(daily_expose))
                pad = np.zeros(MAX_DAY - len(temp))
                temp = np.append(temp, pad)
                
                if (len(doubling_times) == 0):
                    doubling_times = temp
                else:
                    doubling_times = np.c_[doubling_times, temp ]
            
            print ("doubling_times:", doubling_times.shape)
            mid = np.median(doubling_times, axis=1)
            upper = np.percentile(doubling_times, q=75, axis=1)
            lower = np.percentile(doubling_times, q=25, axis=1)
            ax.plot(range(len(mid)), mid, label=self.get_label(_config))
            ax.fill_between(range(len(mid)), upper, lower, alpha=0.3)
            #print ("-----------")
        Plot.set_lims(ax, xmin, xmax, ymin, ymax)
        ax.set_ylabel("Infection doubling days")
        ax.set_xlabel("Days from start")
        if (legend == True):
            ax.legend()
        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)
        return ax
    '''


    def compare_total_infect_vs_social_value(self, nolockdown_config, compare_configs, df, ax=None, outfilename="", legend=False):
        if (ax == None):
            fig, ax = plt.subplots()

        #all_checkin_count = len(df)
        all_checkin_counts = np.bincount((df['posixtime']/24.).astype('int').to_numpy())
        nolockdown_results = np.load(self.result_dir+"_".join([str(x) for x in nolockdown_config])+".npy", allow_pickle=True)

        temp_infects = np.array([])
        for us, _, c_with in nolockdown_results:
            expose_times, _ = Plot.get_expose_recover_times(us)
            temp_infects = np.append(temp_infects, len(expose_times))
        mid_infects_nolockdown = np.median(temp_infects)
        
        for name, _cfgs in compare_configs:
            data = np.array([])
            print (name)
            for _cfg in _cfgs:
                filename = "_".join([str(x) for x in _cfg])
                results = np.load(self.result_dir+filename+".npy", allow_pickle=True)

                temp_infects, temp_checkins = np.array([]), np.array([])
                for user_dict, _counts_skip, counts_with in results:
                    counts_with = counts_with[:150]
                    #counts_with = 100*np.divide(_counts_with, all_checkins, out=np.zeros_like(_counts_with), where=all_checkins!=0)
                    
                    expose_times, _ = Plot.get_expose_recover_times(user_dict)
                    temp_infects = np.append(temp_infects, 100*(1- len(expose_times)/float(mid_infects_nolockdown)))
                    #print ("all_checkin_counts:", all_checkin_counts)
                    #print ("counts_with:", counts_with)
                    temp_checkins = np.append(temp_checkins, 100*sum(counts_with)/float( sum(all_checkin_counts[:150]) ))

                temp = np.array([np.median(temp_checkins), np.percentile(temp_checkins, 25), np.percentile(temp_checkins, 75),\
                                 np.median(temp_infects),  np.percentile(temp_infects, 25), np.percentile(temp_infects, 75)])
                if (len(data) == 0):
                    data = temp
                else:
                    data = np.vstack(( data, temp ))
            
            data = data[data[:,0].argsort()]
            x, xlower, xupper, y, ylower, yupper = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
            ax.errorbar(x, y, xerr=[x-xlower, xupper-x], yerr=[y-ylower, yupper-y], marker='o', alpha=0.8, label=name)
            #ax.plot(data[:,0], data[:,3], marker='o', label=name)
            #break

        ax.set_xlabel('Social Value')
        ax.set_ylabel('Health Value')
        if (legend == True):
            ax.legend()
        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)
        return ax
    
    def tracking_popular_users(self, config, df, ax=None, outfilename="", xmin=None, xmax=None, ymin=None, ymax = None, legend=False):
        if (ax == None):
            fig, ax = plt.subplots()
        
        for top_percentile in [1, 5, 100]:
            rank = int(math.ceil(self.nusers*top_percentile/100.))
            xf = df.groupby('userId').count()[['venueId']]
            sorted_xf = xf.sort_values('venueId', ascending=False)
            S = sorted_xf.index.tolist()[:rank]
            

            filename = "_".join([str(x) for x in config])
            results = np.load(self.result_dir+filename+".npy", allow_pickle=True)
            active_cases = np.array([])
            for user_dict, _, _ in results:
                daily_expose, daily_recover = np.zeros(MAX_DAY), np.zeros(MAX_DAY)
                for uid in user_dict.keys():
                    if (uid in S):
                        if (user_dict[uid].expose_time > 0):
                            expose_day = int((user_dict[uid].expose_time-self.start_time)/24.)
                            recover_day = int((user_dict[uid].recover_time-self.start_time)/24.)
                            daily_expose[expose_day] += 1 
                            daily_recover[recover_day] += 1
                daily_expose = 100*daily_expose/float(len(S))
                daily_recover = 100*daily_recover/float(len(S))
                temp = Plot.get_active_cases(daily_expose, daily_recover)
                if (len(active_cases) == 0):
                    active_cases = temp
                else:
                    active_cases = np.c_[active_cases, temp ]
            #print ("active_cases:", active_cases.shape)
            mid = np.around(np.median(active_cases, axis=1), 2)
            max_day = np.argmax(mid)
            #max_infect = np.max(mid)
            #max_day = np.min(np.where(mid>max_infect*0.5)[0])
            
            print (top_percentile, "max_day", max_day, "max val:", max(mid))
            upper = np.percentile(active_cases, q=75, axis=1)
            lower = np.percentile(active_cases, q=25, axis=1)
            
            label = "Most active "+str(top_percentile)+" % people"
            if (top_percentile == 100):
                label="Whole population"
            _p = ax.plot(range(len(mid)), mid, label=label)
            ax.axvline(x=max_day, ls='--', color=_p[0].get_color())
            ax.fill_between(range(len(mid)), upper, lower, alpha=0.3)
            #print ("-----------")
        Plot.set_lims(ax, xmin, xmax, ymin, ymax)
        ax.set_ylabel("% of tracked people active")
        ax.set_xlabel("Days from start")
        if (legend == True):
            ax.legend()
        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)
        return ax


    def tracking_popular_venues(self, config, df, ax=None, outfilename="", xmin=None, xmax=None, ymin=None, ymax = None, legend=False):
        if (ax == None):
            fig, ax = plt.subplots()
        
        for top_percentile in [0.01, 0.1, 100]:
            nvenues = len(df.venueId.unique())
            rank = int(math.ceil(nvenues*top_percentile/100.))
            xf = df.groupby('venueId').count()[['userId']]
            sorted_xf = xf.sort_values('userId', ascending=False)
            S = sorted_xf.index.tolist()[:rank]
            S1 = df[df['venueId'].isin(S)].copy().reset_index(drop=True).userId.unique()
            print ("after selection # users:", len(S1), "# venues:", len(S))

            filename = "_".join([str(x) for x in config])
            results = np.load(self.result_dir+filename+".npy", allow_pickle=True)
            active_cases = np.array([])
            for user_dict, _, _ in results:
                daily_expose, daily_recover = np.zeros(MAX_DAY), np.zeros(MAX_DAY)
                for uid in user_dict.keys():
                    if (uid in S1):
                        if (user_dict[uid].expose_time > 0):
                            expose_day = int((user_dict[uid].expose_time-self.start_time)/24.)
                            recover_day = int((user_dict[uid].recover_time-self.start_time)/24.)
                            daily_expose[expose_day] += 1 
                            daily_recover[recover_day] += 1
                daily_expose = 100*daily_expose/float(len(S1))
                daily_recover = 100*daily_recover/float(len(S1))
                temp = Plot.get_active_cases(daily_expose, daily_recover)
                if (len(active_cases) == 0):
                    active_cases = temp
                else:
                    active_cases = np.c_[active_cases, temp ]
            #print ("active_cases:", active_cases.shape)
            mid = np.around(np.median(active_cases, axis=1), 2)
            max_day = np.argmax(mid)
            #max_infect = np.max(mid)
            #max_day = np.min(np.where(mid>max_infect*0.5)[0])
            
            print (top_percentile, "max_day", max_day, "max val:", max(mid))
            upper = np.percentile(active_cases, q=75, axis=1)
            lower = np.percentile(active_cases, q=25, axis=1)
            
            label = "Tracking people ("+ str(int(np.around(100*len(S1)/self.nusers)))+"%) who visited at least a "+str(top_percentile)+"% ("+ str(len(S))+") most popular venues."
            print ("label:", label)
            label = "Tracking people who visited at least a "+str(top_percentile)+"% most popular venues."
            if (top_percentile == 100):
                label="Whole population"
            _p = ax.plot(range(len(mid)), mid, label=label)
            #ax.axvline(x=max_day, ls='--', color=_p[0].get_color())
            ax.fill_between(range(len(mid)), upper, lower, alpha=0.3)
            #print ("-----------")
        Plot.set_lims(ax, xmin, xmax, ymin, ymax)
        ax.set_ylabel("% of tracked people actively infected")
        ax.set_xlabel("Days from start")
        if (legend == True):
            ax.legend()
        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)
        return ax

    def popular_venue_infection_vs_checkin(self, config, df, ax=None, outfilename="", xmin=None, xmax=None, ymin=None, ymax = None, legend=False):
        filename = "_".join([str(x) for x in config])
        results = np.load(self.result_dir+filename+".npy", allow_pickle=True)
        venues = df.venueId.unique().tolist()
        #venue_dict = {k: {'infect_count':np.zeros(10), 'checkin_count':len(df.loc[df['venueId'] == k])}\
        #         for k, v in zip(venues, np.zeros(len(venues), dtype='int'))}
        venue_dict = {k: np.zeros(10) for k, v in zip(venues, np.zeros(len(venues), dtype='int'))}
        print (len(venue_dict))
        
        #exit(0)
        it_index = 0
        for user_dict, _, _ in results:
            for uid in user_dict.keys():
                if (user_dict[uid].expose_time > 0 and user_dict[uid].infect_from_venue != None):
                    venue_dict[user_dict[uid].infect_from_venue][it_index] += 1
            it_index += 1
            
        arr = np.array([])
        for k, v in venue_dict.items():
            if (np.median(v) > 0):
                arr = np.append (arr, np.median(v) )
        
        arr = np.sort(arr)
        carr = sum(arr) - np.cumsum(arr)
        #print (arr)
        ax.plot(carr)
        ax.set_xscale('log'); ax.set_yscale('log')
        '''
        #ax.scatter(data_pts[:,0], data_pts[:,1])
        hist, bin_edges = np.histogram(arr, bins = 100)
        #hist = hist + 1
        x = np.log10(bin_edges[:-1])
        y = np.log10(hist)
        ax.scatter(x, y) #label=label)
        
        #Plot.set_lims(ax, xmin, xmax, ymin, ymax)
        ax.set_xlabel("Infect count (log)")
        ax.set_ylabel("# of such venues (log)")
        '''
        if (legend == True):
            ax.legend()
        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)
        return ax


'''
    def pick_active_case_day(self, ax=None, outfilename="", xmin=None, xmax=None, ymin=None, ymax = None):
        if (ax == None):
            fig, ax = plt.subplots()
        mid_pick_days = np.array([])
        upper_pick_days = np.array([])
        lower_pick_days = np.array([])
        xs = []
        for _config in self.configs:
            filename = "_".join([str(x) for x in _config])
            results = np.load(self.result_dir+filename+".npy", allow_pickle=True)
            _pick_days = np.array([])
            for res in results:
                infect_times, remove_times = res
                daily_infect = self.daily_count(infect_times)*100/float(self.nusers)
                daily_remove = self.daily_count(remove_times)*100/float(self.nusers)
                active_cases = Plot.get_active_cases(daily_infect, daily_remove)
                _pick_days = np.append(_pick_days, np.argmax(active_cases))
            mid_pick_days = np.append(mid_pick_days, np.median(_pick_days))
            upper_pick_days = np.append(upper_pick_days, np.percentile(_pick_days, q=75))
            lower_pick_days = np.append(lower_pick_days, np.percentile(_pick_days, q=25))
            xs.append(_config[self.label_config.config_index])
        
        ax.plot(xs, mid_pick_days, marker='o', label="")
        ax.fill_between(xs, upper_pick_days, lower_pick_days, alpha=0.2)
        Plot.set_lims(ax, xmin, xmax, ymin, ymax)
        ax.set_ylabel("Day to pick active cases")
        ax.set_xlabel(self.label_config.xlabel)
        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)

     
    def total_infect(self, ax=None, outfilename="", xmin=None, xmax=None, ymin=None, ymax = None):
        if (ax == None):
            fig, ax = plt.subplots()
        mid = np.array([])
        upper = np.array([])
        lower = np.array([])
        xs = []
        for _config in self.configs:
            filename = "_".join([str(x) for x in _config])
            results = np.load(self.result_dir+filename+".npy", allow_pickle=True)
            _total_infects = np.array([])
            for res in results:
                infect_times, remove_times = res
                _total_infects = np.append(_total_infects, len(infect_times)*100/float(self.nusers))
            mid = np.append(mid, np.median(_total_infects))
            upper = np.append(upper, np.percentile(_total_infects, q=75))
            lower = np.append(lower, np.percentile(_total_infects, q=25))
            xs.append(_config[self.label_config.config_index])
        
        ax.plot(xs, mid, marker='o', label="")
        ax.fill_between(xs, upper, lower, alpha=0.2)
        Plot.set_lims(ax, xmin, xmax, ymin, ymax)
        ax.set_ylabel("% of users finally infected")
        ax.set_xlabel(self.label_config.xlabel)
        if (outfilename != ""):
            Plot.saveplotfile(fig, outfilename)
'''



############ Unit tests ################
def unit_test_venue_cleaning():
    c = Configs(4, 0.75, 48, 0.7, 7*24, 1*24, 21*24, 1*24, 4, 24, 0, 0)
    test_cases = [
                    #last_infect_time, curtime, expected_last_infect_time
                    [10, 12, 10],
                    [10, 25, 0],
                    [10, 47, 0],
                    [25, 26, 25],
                    [25, 49, 0],
                    [10, 48, 0],
                 ]
    for last_infect_time, curtime, expected_last_infect_time in test_cases:
        v = Venue(c)
        v.set_last_infect_time(last_infect_time)
        #v.update_last_clean_time(50)
        if (v.get_last_infect_time(curtime) != expected_last_infect_time):
            print ("Error", last_infect_time, curtime, expected_last_infect_time)
    print ("Test done")

def unit_test_infecting_user_stored_in_venue():
    c = Configs(1, 1.0, 48, 0, 7, 0, 21, 0, 3, 0, 1, 0, 0, 0)
    userid = 100
    u = User(c)
    v = Venue(c)
    v.infect_venue(10, userid)
    print ("last infect time:", v.get_last_infect_time(12), "v:", v)
    print ("last infect time:", v.get_last_infect_time(58.1), "v:", v)

def unit_test_simulation():
    c = Configs(1, 1.0, 1, 0.0, 0, 0, 0, 0)
    c._reset_interval = 48
    c._infect_prob = 1.0
    c._asymp_prob = 0.0
    c._DBG = True

    c.print("Config values:", c)
    data = [
             [100, "R1", 1.0],
             [101, "R1", 47.0],
             [101, "R1", 200.0],
             [102, "R1", 211.0],
             [103, "R1", 240.0],
             
             [100, "R2", 2.0],
             [105, "R2", 4.0],
             
           ]

    df = pd.DataFrame(data, columns =['userId', 'venueId', 'posixtime'])
    df.to_csv('/tmp/dummy.csv')
    sim = MobilitySimulator('/tmp/dummy.csv', start_time=0, nusers=df.userId.unique(),\
                                nvenues=df.venueId.unique(), ncheckins=len(df), config=c)
    user_dict = sim.run()
    print ("user_dict:\n", user_dict)
    print ("venue_dict:\n", sim.venue_dict)


def unit_test_user():
    c = Configs(4, 1.0, 48, 0, 0, 0, 0, 0)
    c._asymp_prob = 0.0
    test_cases = [
                    #curtime action(get/next state) expected_state
                    [
                        [10, "GET", STATE_S],
                        [11, "INFECT", None],
                        [12, "GET", STATE_E],
                        [13, "GET", STATE_E],
                        [16, "GET", STATE_I],
                        [20, "GET", STATE_R],
                    ],
                ]
    print (c)
    u = User(c)
    
    u.expose(10, is_seed=True)
    print (u)
    for t in [11, 153, 154, 155, 273, 274, 275, 442, 443]:
        print (t, "state:[", state_str[u.get_state(t)],"]", u)

if __name__ == "__main__":
    #unit_test_user_incubation()
    #unit_test_infecting_user()
    #unit_test_infecting_user_simulation()
    unit_test_simulation()
    #unit_test_user()
