import pandas as pd
import numpy as np
import multiprocessing
import networkx as nx
from scipy import spatial
from joblib import Parallel, delayed
import copy, os

from mobility_simulator import *

LOCKDOWN_1 = 1
LOCKDOWN_3 = 3
LOCKDOWN_5 = 5

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['legend.numpoints'] = 1
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


class GraphSimulator:
    def __init__(self, graph, config):
        self.config = config
        self.graph = graph
        self.user_dict = {}
        self.venue_dict = {}
        for idx, data in graph.nodes(data=True):
            if (data['ntype'] == 'user'):
                self.user_dict[idx] = User(config)
            elif (data['ntype'] == 'venue'):
                self.venue_dict[idx] = Venue(config)
    
    def select_seeds(self):
        _seeds = np.random.choice(list(self.user_dict.keys()), size=self.config._num_seeds, replace=False)
        for __s in _seeds:
            try:
                self.user_dict[__s].expose(1, is_seed=True)
            except Exception:
                print ("** exception for __s:", __s, "seeds:", _seeds)
                exit(0)
        return _seeds

    def process_users_one_round(self, infected_users, curtime):
        new_venue_infect, new_user_remove = [], []
        today = int(curtime/24.)
        for userid in infected_users:
            state = self.user_dict[userid].get_state(curtime)
            
            if (state != STATE_I):
                if (state == STATE_R):
                    new_user_remove.append(userid)
                continue
            
            for ngh_venue in self.graph.neighbors(userid):
                w = self.graph.edges[userid, ngh_venue]['weight']
                prob = min (1, w/150.)
                
                if (np.random.uniform() <= prob):
                    self.venue_dict[ngh_venue].infect_venue(curtime, userid)
                    new_venue_infect.append(ngh_venue)
        
        return new_venue_infect, new_user_remove
    

    def process_venues_one_round(self, infected_venues, curtime):
        new_venue_remove, new_user_infect = [], []
        today = int(curtime/24.)
        
        for venueid in infected_venues:
            last_infect_time, _ = self.venue_dict[venueid].get_last_infect_time(curtime)
            if (last_infect_time <= 0):
                new_venue_remove.append(venueid)
                continue
            
            for ngh_user in self.graph.neighbors(venueid):
                state = self.user_dict[ngh_user].get_state(curtime)
                if (state != STATE_S):
                    continue
                
                p = self.config._infect_prob
                
                w = int(self.graph.edges[venueid, ngh_user]['weight'])
                prob = 1 - np.power( (1 - p), w / 150.)

                if (np.random.uniform() <= prob):
                    # user gets infected
                    self.user_dict[ngh_user].expose(curtime)
                    new_user_infect.append(ngh_user)
        
        return new_venue_remove, new_user_infect
    
    def run(self):
        infected_users = np.unique(self.select_seeds()).tolist()
        print("seeds:", infected_users)
        infected_venues = []
        
        today = 1
        yesterday = 0
        
        
        while(True):
            if (today >= 150):
                break;

            new_venue_infect, new_user_remove = self.process_users_one_round(infected_users, today*24)
            new_venue_remove, new_user_infect = self.process_venues_one_round(infected_venues, today*24)

            [ infected_users.remove(u) for u in new_user_remove ]
            [ infected_venues.remove(v) for v in new_venue_remove ]

            [infected_users.append(u) for u in new_user_infect]
            infected_users = np.unique(infected_users).tolist()
            [infected_venues.append(u) for u in new_venue_infect]
            infected_venues = np.unique(infected_venues).tolist()

            today += 1
            if( today != yesterday):
                if (today % 50 == 0):
                    print (today, "len(infected_users):", len(infected_users), "len(infected_venues):", len(infected_venues))
                yesterday = today
            
            if (len(infected_venues) == 0 and len(infected_users) == 0):
                break;
        
        return self.user_dict


def get_cdf_infects(filename, start_time, nusers, is_mobility):
    results = np.load(filename, allow_pickle=True)
    cum_exposed = np.array([])
    for _res in results:
        res = _res
        if (is_mobility == True):
            res = _res[0]
        expose_times, recover_times = Plot.get_expose_recover_times(res)
        
        daily_expose = Plot.daily_count(start_time, expose_times)*100/float(nusers)
        if (len(cum_exposed) == 0):
            cum_exposed = np.cumsum(daily_expose)
        else:
            cum_exposed = np.c_[cum_exposed, np.cumsum(daily_expose) ]
    return cum_exposed

def plot_single_cdf_infects(ax, filename, label, start_time, nusers, is_mobility):
    cum_exposed = get_cdf_infects(filename, start_time, nusers, is_mobility)
    #print ("cum_infects:", cum_infects.shape)
    mid = np.median(cum_exposed, axis=1)
    upper = np.percentile(cum_exposed, q=75, axis=1)
    lower = np.percentile(cum_exposed, q=25, axis=1)
    ax.plot(range(len(mid)), mid, label=label)
    ax.fill_between(range(len(mid)), upper, lower, alpha=0.3)

def plot_single_active_cases(ax, filename, label, start_time, nusers, is_mobility):
    results = np.load(filename, allow_pickle=True)
    active_cases = np.array([])
    for _res in results:
        res = _res
        if (is_mobility == True):
            res = _res[0]
        expose_times, recover_times = Plot.get_expose_recover_times(res)
        daily_expose = Plot.daily_count(start_time, expose_times)*100/float(nusers)
        daily_recover = Plot.daily_count(start_time, recover_times)*100/float(nusers)
        temp = Plot.get_active_cases(daily_expose, daily_recover)
        if (len(active_cases) == 0):
            active_cases = temp
        else:
            active_cases = np.c_[active_cases, temp ]
    #print ("cum_infects:", cum_infects.shape)
    mid = np.median(active_cases, axis=1)
    upper = np.percentile(active_cases, q=75, axis=1)
    lower = np.percentile(active_cases, q=25, axis=1)
    ax.plot(range(len(mid)), mid, label=label)
    ax.fill_between(range(len(mid)), upper, lower, alpha=0.3)

def plot_single_Rt(ax, filename, label, start_time, is_mobility):
    results = np.load(filename, allow_pickle=True)
    Rts = np.array([])
    for _res in results:
        user_dict = _res
        if (is_mobility == True):
            user_dict = _res[0]
        _infects_from_me = np.zeros(MAX_DAY)
        _infected_today = np.zeros(MAX_DAY)
        for uid in user_dict.keys():
            if (user_dict[uid].expose_time > 0):
                day = int((user_dict[uid].expose_time - start_time)/24.)
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
        
        if (len(Rts) == 0):
            Rts = Rt
        else:
            Rts = np.c_[Rts, Rt]
    mid = np.median(Rts, axis=1)
    upper = np.percentile(Rts, q=75, axis=1)
    lower = np.percentile(Rts, q=25, axis=1)
    ax.plot(range(len(mid)), mid, label=label)
    ax.fill_between(range(len(mid)), upper, lower, alpha=0.3)


def plot_no_lockdown(city, PLOT_DIR):
    def compare_cdf_infects(city, PLOT_DIR):
        fig, ax = plt.subplots()
        _config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]; filename = "_".join([str(x) for x in _config])
        
        ## Mobility simulation
        nusers = len(pd.read_csv("../data/"+city+"/checkin.csv").userId.unique())
        plot_single_cdf_infects(ax, "../result/mobility/"+city+"/"+filename+".npy", "Mobility simulation", 0, nusers, True)
        #plot_single_cdf_infects(ax, "../result/graph/"+city+"/frnd_graph_contagion.npy", "Friendship Graph", 0, nusers, False)
        
        _config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]; filename = "_".join([str(x) for x in _config])
        plot_single_cdf_infects(ax, "../result/graph/"+city+"/"+filename+"_b.npy", "Contact graph simulation", 0, nusers, False)

        Plot.set_lims(ax, xmin=0, xmax=150, ymin=0, ymax=None)
        ax.set_xlabel("# rounds / days from start")
        ax.set_ylabel("% of people infected")
        #ax.legend()
        
        Plot.saveplotfile(fig, PLOT_DIR+"compare_cdf_infects")
        
        Plot.plot_legends(ax, 2, PLOT_DIR+"compare_legends")
    
    def get_infect_users(user_dict):
        _infects = [[] for x in range(200)]
        for uid in user_dict.keys():
            if (user_dict[uid].expose_time > 0):
                _infects[int(user_dict[uid].expose_time/24.)].append(uid)
        # compute cumulative
        for ix in range(1, len(_infects)):
            _infects[ix] = set(_infects[ix-1]).union(set(_infects[ix]))
        return _infects

    def plot_jaccard_users(city, PLOT_DIR):
        fig, ax = plt.subplots()
        _config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]
        nusers = len(pd.read_csv("../data/"+city+"/checkin.csv").userId.unique())
        mobility_filename = "../result/mobility/"+city+"/"+"_".join([str(x) for x in _config])+".npy"
        graph_filename = "../result/graph/"+city+"/"+"_".join([str(x) for x in _config])+"_b.npy"

        mobility_results = np.load(mobility_filename, allow_pickle=True)
        graph_results = np.load(graph_filename, allow_pickle=True)
        sims = np.array([])
        
        
        for ix in range(len(graph_results)):
            if (ix >= len(mobility_results)):
                break
            m_infects = get_infect_users(mobility_results[ix][0])
            g_infects = get_infect_users(graph_results[ix])

            temp = []
            for d in range(200):
                union = float(len(set(m_infects[d]).union(set(g_infects[d]))))
                if (union > 0):
                    sim = len(set(m_infects[d]).intersection(set(g_infects[d]))) / union
                else:
                    sim = 0
                temp.append(sim*100.)
            if (len(sims) == 0):
                sims = temp
            else:
                sims = np.c_[sims, temp ]
    
        mid = np.median(sims, axis=1)
        upper = np.percentile(sims, q=75, axis=1)
        lower = np.percentile(sims, q=25, axis=1)
        ax.plot(range(len(mid)), mid)
        ax.fill_between(range(len(mid)), upper, lower, alpha=0.3)
        

        Plot.set_lims(ax, xmin=0, xmax=150, ymin=0, ymax=None)
        ax.set_xlabel("# rounds / days from start")
        ax.set_ylabel("% people infected in common")
        #ax.legend()

        Plot.saveplotfile(fig, PLOT_DIR+"compare_user_similarity")


    def compare_active_cases(city, PLOT_DIR):
        fig, ax = plt.subplots()
        _config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]; filename = "_".join([str(x) for x in _config])        
        ## Mobility simulation
        nusers = len(pd.read_csv("../data/"+city+"/checkin.csv").userId.unique())
        plot_single_active_cases(ax, "../result/mobility/"+city+"/"+filename+".npy", "Mobility simulation", 0, nusers, True)
        #plot_single_active_cases(ax, "../result/graph/"+city+"/frnd_graph_contagion.npy", "Friendship Graph", 0, nusers, False)
        
        _config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]; filename = "_".join([str(x) for x in _config])
        plot_single_active_cases(ax, "../result/graph/"+city+"/"+filename+"_b.npy", "Contact graph simulation", 0, nusers, False)

        Plot.set_lims(ax, xmin=0, xmax=150, ymin=0, ymax=None)
        ax.set_xlabel("# rounds / days from start")
        ax.set_ylabel("% of people still infected")
        Plot.saveplotfile(fig, PLOT_DIR+"/compare_active_cases")
        

    compare_cdf_infects(city, PLOT_DIR)
    compare_active_cases(city, PLOT_DIR)
    plot_jaccard_users(city, PLOT_DIR)


def single_experiment(graph, c):
    sim = GraphSimulator(graph, c)
    return sim.run()

def run_no_lockdown(city):
    result_dirname = "../result/graph/"+city+"/"; ensure_dir(result_dirname)
    graph = nx.read_gpickle( "../data/"+city+"/bipartite_graph" )
    
    skip_prob = 0.0
    _config = [10, 0.75, 48, 0.35, 0, 0, 0, 0]
    c = Configs(*_config)
    c._city = city
    filename = "_".join([str(x) for x in _config])+"_b"
    
    iteration = 20; n_jobs = 10
    result = Parallel(n_jobs=n_jobs)(delayed(single_experiment)(graph, c) for _ in range(iteration))

    print("saving ", filename, len(result))
    np.save(result_dirname+filename, result)


if __name__ == "__main__":
    city = "london"
    run_no_lockdown(city)
    
    
    PLOT_DIR = "../plot/graph/"+city+"/"; ensure_dir(PLOT_DIR)
    plot_no_lockdown(city, PLOT_DIR)
    plt.show()
    
    
    
