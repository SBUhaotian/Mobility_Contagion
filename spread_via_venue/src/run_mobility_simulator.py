import pandas as pd
import numpy as np
from mobility_simulator import *
import copy
import os
import sys
import random 
from scipy import ndimage

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


def compute_avg_contact(CITY, start_time, nusers):
    print ("compute_avg_contact")
    config = Configs(4, 0.75, 2*24, 0.7, 7*24, 1*24, 21*24, 1*24, 6*24, 1*24, 0, 0, 0, 0)
    avg_contact_perday_peruser = config.compute_avg_contact('../data/'+CITY+'.csv', start_time, nusers)
    print (avg_contact_perday_peruser)
    for period in [0.65*(11-6) + 0.35* (24-6), 11-6, 7]:
        print ("period:", period, avg_contact_perday_peruser*period)

def convert(configs):
    for c in configs:
        c[1] = get_infect(c[1])
    print("configs:", configs)
    return configs

def get_mobility_restrictions():
    # https://www.google.com/covid19/mobility/
    df = pd.read_csv("../data/Global_Mobility_Report.csv")
    _restrictions = df[df['sub_region_2'].str.contains("New York County", na=False)]\
                [ ['retail_and_recreation_percent_change_from_baseline'] ].to_numpy().flatten()
    #restrictions = restrictions - np.max(restrictions)
    restrictions = []
    for r in _restrictions:
        if (r >= 0 ):
            restrictions.append(0)
        else:
            restrictions.append( (-1* r/100.)/2. )
    return np.asarray(restrictions)


def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def single_experiment(df, start_time, nusers, nvenues, ncheckins, c):
    if (c._lockdown_num == 5):
        daily_expose = np.zeros(MAX_DAY)
        daily_recover = np.zeros(MAX_DAY)
        nusers = 1/float(c._lockdown_arg)*nusers
        all_users = df.userId.unique().tolist()
        user_divs = partition(all_users, c._lockdown_arg)
        _df = df.copy()
        for ii in range(c._lockdown_arg):
            selected_users = user_divs[ii]
            #continue
            __df = _df[_df['userId'].isin(selected_users)].reset_index(drop=True)
            sim = MobilitySimulator(__df, start_time, nusers, nvenues, ncheckins, config=c)
            u_dict, x, checkin_counts =  sim.run()
            expose_times, recover_times = Plot.get_expose_recover_times(u_dict)
            _daily_expose = Plot.daily_count(0, expose_times, 24)
            _daily_recover = Plot.daily_count(0, recover_times, 24)
            
            daily_expose += _daily_expose
            daily_recover += _daily_recover
        return None, [daily_expose, daily_recover], None
    
    else:
        sim = MobilitySimulator(df, start_time, nusers, nvenues, ncheckins, config=c)
        return sim.run()


if __name__ == "__main__":

    #num_seeds, infect_prob, reset_interval, asymp_prob, lockdown_num, lockdown_arg, lockdown_offset, lockdown_len
    configs = {
                "LOCKDOWN_0":
                [
                    #[10, 0.85, 24*2, 0.35, 0, 0, 0, 0],
                    [10, 0.75, 24*2, 0.35, 0, 0, 0, 0],
                    [10, 0.5, 24*2, 0.35, 0, 0, 0, 0],
                    [10, 0.25, 24*2, 0.35, 0, 0, 0, 0],
                    #[10, 0.15, 24*2, 0.35, 0, 0, 0, 0],
                ],
                "LOCKDOWN_100":
                [
                    [1, 0.75, 24*2, 0.35, 0, 0, 0, 0],
                    [5, 0.75, 24*2, 0.35, 0, 0, 0, 0],
                    [20, 0.75, 24*2, 0.35, 0, 0, 0, 0],
                ],
                "LOCKDOWN_101":
                [
                    [10, 0.75, 24*2, 0.15, 0, 0, 0, 0],
                    [10, 0.75, 24*2, 0.35, 0, 0, 0, 0],
                    [10, 0.75, 24*2, 0.75, 0, 0, 0, 0],
                ],
                "LOCKDOWN_1": # Random subsample of checkins
                [
                    [10, 0.75, 24*2, 0.35, 1, 0.1, 0, 0],
                    [10, 0.75, 24*2, 0.35, 1, 0.2, 0, 0],
                    [10, 0.75, 24*2, 0.35, 1, 0.4, 0, 0],
                    [10, 0.75, 24*2, 0.35, 1, 0.6, 0, 0],
                    [10, 0.75, 24*2, 0.35, 1, 0.8, 0, 0],
                    [10, 0.75, 24*2, 0.35, 1, 0.9, 0, 0],
                ],
                "LOCKDOWN_2": # Close popular venues
                [
                    ##[10, 0.75, 24*2, 0.35, 2, 0.01, 0, 0],
                    [10, 0.75, 24*2, 0.35, 2, 0.1, 0, 0],
                    [10, 0.75, 24*2, 0.35, 2, 1, 0, 0],
                    #[10, 0.75, 24*2, 0.35, 2, 10, 0, 0],
                    #[10, 0.75, 24*2, 0.35, 2, 20, 0, 0],
                ],                
                "LOCKDOWN_15": # Close popular venues - scaling
                [
                    [10, 0.75, 24*2, 0.35, 15, 0, 0, 0],
                    [10, 0.75, 24*2, 0.35, 15, 0.1, 0, 0],
                    [10, 0.75, 24*2, 0.35, 15, 0.5, 0, 0],
                    [10, 0.75, 24*2, 0.35, 15, 1, 0, 0],
                    [10, 0.75, 24*2, 0.35, 15, 10, 0, 0],
                ],
                "LOCKDOWN_3": # Isolate popular users
                [
                    [10, 0.75, 24*2, 0.35, 3, 0.1, 0, 0],
                    [10, 0.75, 24*2, 0.35, 3, 1, 0, 0],
                    [10, 0.75, 24*2, 0.35, 3, 10, 0, 0],
                    [10, 0.75, 24*2, 0.35, 3, 20, 0, 0],
                ],
                "LOCKDOWN_16": # Close popular venues - scaling
                [
                    [10, 0.75, 24*2, 0.35, 16, 0, 0, 0],
                    [10, 0.75, 24*2, 0.35, 16, 1, 0, 0],
                    [10, 0.75, 24*2, 0.35, 16, 10, 0, 0],
                    [10, 0.75, 24*2, 0.35, 16, 20, 0, 0],
                ],
                "LOCKDOWN_4": # Cleaning
                [
                    [10, 0.75, 24*2, 0.35, 4, 6, 0, 0],
                    [10, 0.75, 24*2, 0.35, 4, 12, 0, 0],
                    [10, 0.75, 24*2, 0.35, 4, 48, 0, 0],
                ],
                "LOCKDOWN_5": # Different user groups operate in cycle
                [
                    [10, 0.75, 24*2, 0.35, 5, 1, 0, 0],
                    [10, 0.75, 24*2, 0.35, 5, 2, 0, 0],
                    [10, 0.75, 24*2, 0.35, 5, 3, 0, 0],
                    [10, 0.75, 24*2, 0.35, 5, 4, 0, 0],
                ],
                
                "LOCKDOWN_6": # Random subsamples of venues are discarded
                [
                    [10, 0.75, 24*2, 0.35, 6, 1, 0, 0],
                    [10, 0.75, 24*2, 0.35, 6, 10, 0, 0],
                    [10, 0.75, 24*2, 0.35, 6, 20, 0, 0],
                    [10, 0.75, 24*2, 0.35, 6, 40, 0, 0],
                    [10, 0.75, 24*2, 0.35, 6, 60, 0, 0],
                ],
                "LOCKDOWN_7": # Random subseamples of users are discarded
                [
                    [10, 0.75, 24*2, 0.35, 7, 1, 0, 0],
                    [10, 0.75, 24*2, 0.35, 7, 10, 0, 0],
                    [10, 0.75, 24*2, 0.35, 7, 20, 0, 0],
                    [10, 0.75, 24*2, 0.35, 7, 40, 0, 0],
                    [10, 0.75, 24*2, 0.35, 7, 60, 0, 0],
                ],

                "LOCKDOWN_81": # skip checkins - vary lockdown offset
                [
                    [10, 0.75, 24*2, 0.35, 8, 0.8, 5, 15],
                    [10, 0.75, 24*2, 0.35, 8, 0.8, 10, 15],
                    [10, 0.75, 24*2, 0.35, 8, 0.8, 15, 15],
                ],
                "LOCKDOWN_82": # skip checkins - vary lockdown length
                [
                    [10, 0.75, 24*2, 0.35, 8, 0.8, 10, 7],
                    [10, 0.75, 24*2, 0.35, 8, 0.8, 10, 15],
                    [10, 0.75, 24*2, 0.35, 8, 0.8, 10, 30],
                    [10, 0.75, 24*2, 0.35, 8, 0.8, 10, 45],
                    [10, 0.75, 24*2, 0.35, 8, 0.8, 10, 60],
                ],
                "LOCKDOWN_83": # skip checkins - vary lockdown skip probability
                [
                    [10, 0.75, 24*2, 0.35, 8, 0.2, 10, 15],
                    [10, 0.75, 24*2, 0.35, 8, 0.5, 10, 15],
                    [10, 0.75, 24*2, 0.35, 8, 0.8, 10, 15],
                ],

                "LOCKDOWN_9": # Lockdown data from Google mobility report.
                [
                    #[10, 2, 24*2, 0.35, 9, 0, 0, 200],
                    #[10, 3, 24*2, 0.35, 9, 0, 0, 200],
                    [10, 4, 24*2, 0.35, 9, 0, 0, 200],
                ],
                "LOCKDOWN_11": # Scalling by taking random subsamples of users
                [
                    # Percentages are controlled below
                    [10, 0.75, 24*2, 0.35, 11, None, 0, 0], # Take 10 percent of users.
                    [10, 0.5, 24*2, 0.35, 11, None, 0, 0],
                    [10, 0.25, 24*2, 0.35, 11, None, 0, 0],
                ],
              }
    num_seeds_index, infect_prob_index, reset_interval_index, asymp_prob_index, \
         lockdown_num_index, lockdown_arg_index, lockdown_offset_index, lockdown_len_index = list(range(8))
    
    if (len(sys.argv) <= 2):
        print ("Usage to execute the model: python3 run_mobility_simulator.py nyc run")
        print ("Usage to plot the model: python3 run_mobility_simulator.py nyc plot 0")
        exit(0)

    CITY = sys.argv[1]
    df = pd.read_csv("../data/"+CITY+"/checkin.csv")
    start_time, nusers, nvenues, ncheckins = df.iloc[0]['posixtime'], len(df.userId.unique()), len(df.venueId.unique()), len(df)


    print ("----------Simulating city:"+CITY+"---------------")
    result_dirname = "../result/mobility/"+CITY+"/"; ensure_dir(result_dirname)
    plot_dirname = "../plot/mobility/"+CITY+"/"; ensure_dir(plot_dirname)
    
    mode = sys.argv[2]
    np.random.seed(0)
    if (mode == 'plot'):
        lockdown_num = int(sys.argv[3])
        print ("--------------Lockdown num:", lockdown_num, "-------------------")
        # 51 - Compare social and health value
        # 61 - Tracking popular users
        # 62 - Tracking users who visited at least a popular venue
        
        name = "LOCKDOWN_"+str(lockdown_num)
        #if (CITY in ['nyc', 'london']):
        #    xmax = 120
        #else:
        xmax = 120
        
        if (lockdown_num in [0, 100, 101]):
            if (lockdown_num == 0):
                idx = infect_prob_index; prefix = "$\\beta=$"
            elif (lockdown_num == 100):
                idx = num_seeds_index; prefix = "# seeds="
            elif (lockdown_num == 101):
                idx = asymp_prob_index; prefix = "$\\gamma  $="
            label_config = LabelConfig(idx, prefix=prefix)
            p = Plot(result_dirname, configs[name], label_config, start_time, nusers)
            
            ax = p.cdf_exposed([], N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"cdf_infected")
            p.active_cases([], N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"active_cases")
            p.new_cases([], N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"new_infected")
            p.plot_growth_factor([], N=7, xmin=0, xmax=xmax, outfilename=plot_dirname+name+"growth_factor")
            Plot.plot_legends(ax, 3, outfilename=plot_dirname+name+"_legends")
    
        elif (lockdown_num in list(range(8)) or lockdown_num == 12):
            prefix=""; suffix=""
            if (lockdown_num in [6, 7]):
                suffix="%"
            elif (lockdown_num in [2, 3]):
                suffix="%"
            elif (lockdown_num == 4):
                suffix = "hr"
            elif (lockdown_num == 5):
                prefix = "# groups="
            
            label_config = LabelConfig(lockdown_arg_index, prefix=prefix, suffix=suffix)
            p = Plot(result_dirname, configs[name], label_config, start_time, nusers)
            nolockdown_config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]
            #nolockdown_config = []

            ax = p.cdf_exposed(nolockdown_config, N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"cdf_infected")
            p.active_cases(nolockdown_config, N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"active_cases")
            p.plot_growth_factor(nolockdown_config, N=7, xmin=0, xmax=xmax, outfilename=plot_dirname+name+"growth_factor")
            p.new_cases(nolockdown_config, N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"new_infected")
            
            if (lockdown_num != 5):
                p.social_value(df, nolockdown_config, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"social_value")
            Plot.plot_legends(ax, 4, outfilename=plot_dirname+name+"_legends")

        elif (lockdown_num == 10):
            pass

        elif (lockdown_num == 11):
            fig, ax = plt.subplots()
            ## Plot a curve for each lockdown_arg_index
            label_config = LabelConfig(lockdown_arg_index)
            nolockdown_config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]
            
            for _config in configs[name]:
                nusers = len(df.userId.unique())
                _tot_infects = np.array([])
                percentages = [25, 50, 75, 90, 100]
                for percentage in percentages:
                    num_users = int(nusers*percentage/100.)
                    print(num_users)
                    #print ("num_users:", num_users)
                    _myconfig = copy.deepcopy(_config); _myconfig[lockdown_arg_index] = percentage
                    #p = Plot(result_dirname, [_myconfig], label_config, start_time, num_users)
                    #p.cdf_exposed([], ax=ax, xmin=0, xmax=xmax, ymin=0, ymax=100, legend=True)
                    filename = "_".join([str(x) for x in _myconfig])
                    results = np.load(result_dirname+filename+".npy", allow_pickle=True)
                    total_infected = np.array([])
                    for res in results:
                        expose_times, _ = Plot.get_expose_recover_times(res[0])
                        #print (percentage, len(expose_times), num_users)
                        total_infected = np.append(total_infected, 100*len(expose_times)/float(num_users))
                    #print (total_infected)
                    if (len(_tot_infects) == 0):
                        _tot_infects = total_infected
                    else:
                        _tot_infects = np.c_[_tot_infects, total_infected]
                mid = np.median(_tot_infects, axis=0)
                #print (mid.shape, len(percentages), _tot_infects.shape)
                upper = np.percentile(_tot_infects, q=75, axis=0)
                lower = np.percentile(_tot_infects, q=25, axis=0)
                ax.plot(percentages, mid, label="$\\beta$="+str(_config[infect_prob_index]))
                ax.fill_between(percentages, upper, lower, alpha=0.3)
            ax.set_ylabel("% of users infected")
            ax.set_xlabel("% users sampled")
            Plot.set_lims(ax, xmin=None, xmax=None, ymin=0, ymax=None)
            #ax.legend()
            Plot.saveplotfile(fig, plot_dirname+name)
            Plot.plot_legends(ax, 3, outfilename=plot_dirname+name+"_legends")

        elif (lockdown_num in [15, 16]): ## Scaling experiments for lockdown 2.
            fig, ax = plt.subplots()
            ## Plot a curve for each lockdown_arg_index
            label_config = LabelConfig(lockdown_arg_index)
            
            for _config in configs[name]:
                nusers = len(df.userId.unique())
                _tot_infects = np.array([])
                percentages = [25, 50, 75, 100]
                for percentage in percentages:
                    num_users = int(nusers*percentage/100.)
                    filename = "_".join([str(x) for x in _config])+"_"+str(percentage)
                    results = np.load(result_dirname+filename+".npy", allow_pickle=True)
                    #print ("filename:", filename, "num_users:", num_users)
                    
                    total_infected = np.array([])
                    for res in results:
                        expose_times, _ = Plot.get_expose_recover_times(res[0])
                        total_infected = np.append(total_infected, 100*len(expose_times)/float(num_users))
                        #print ("\ttotal_infected:", total_infected)
                    if (len(_tot_infects) == 0):
                        _tot_infects = total_infected
                    else:
                        _tot_infects = np.c_[_tot_infects, total_infected]
                mid = np.median(_tot_infects, axis=0)
                
                upper = np.percentile(_tot_infects, q=75, axis=0)
                lower = np.percentile(_tot_infects, q=25, axis=0)
                ax.plot(percentages, mid, marker='o', label="$x$="+str(_config[lockdown_arg_index])+"%")
                ax.fill_between(percentages, upper, lower, alpha=0.3)
            ax.set_ylabel("% of users infected in total")
            ax.set_xlabel("Sample size in % of population")
            Plot.set_lims(ax, xmin=None, xmax=None, ymin=0, ymax=None)
            #ax.legend()
            Plot.saveplotfile(fig, plot_dirname+name)
            Plot.plot_legends(ax, 2, outfilename=plot_dirname+name+"_legends")
            
        elif (lockdown_num == 81): # vary lockdown offset for closing popular venues
            
            label_config = LabelConfig(lockdown_offset_index, prefix="", suffix=" %")
            p = Plot(result_dirname, configs[name], label_config, start_time, nusers)
            
            nolockdown_config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]
            
            ax = p.cdf_exposed(nolockdown_config, N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"cdf_infected")
            p.new_cases(nolockdown_config, N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"new_infected")
            p.active_cases(nolockdown_config, N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"active_cases")
            p.plot_growth_factor(nolockdown_config, N=7, xmin=0, xmax=xmax, outfilename=plot_dirname+name+"growth_factor")
            Plot.plot_legends(ax, 5, outfilename=plot_dirname+name+"_legends")

        elif (lockdown_num == 82): # vary lockdown length for closing popular venues
            label_config = LabelConfig(lockdown_len_index, prefix="", suffix=" days")
            p = Plot(result_dirname, configs["LOCKDOWN_"+str(lockdown_num)], label_config, start_time, nusers)
            nolockdown_config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]
            ax = p.cdf_exposed(nolockdown_config, N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"cdf_infected")
            p.active_cases(nolockdown_config, N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"active_cases")
            p.new_cases(nolockdown_config, N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"new_infected")
            p.plot_growth_factor(nolockdown_config, N=7, xmin=0, xmax=xmax, outfilename=plot_dirname+name+"growth_factor")
        
            Plot.plot_legends(ax, 6, outfilename=plot_dirname+name+"_legends")

        elif (lockdown_num == 83): # vary lockdown length for closing popular venues
            label_config = LabelConfig(lockdown_arg_index, prefix="", suffix="")
            p = Plot(result_dirname, configs["LOCKDOWN_"+str(lockdown_num)], label_config, start_time, nusers)
            nolockdown_config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]
            ax = p.cdf_exposed(nolockdown_config, N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"cdf_infected")
            p.active_cases(nolockdown_config, N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"active_cases")
            p.new_cases(nolockdown_config, N=7, xmin=0, xmax=xmax, ymin=0, outfilename=plot_dirname+name+"new_infected")
            p.plot_growth_factor(nolockdown_config, N=7, xmin=0, xmax=xmax, outfilename=plot_dirname+name+"growth_factor")
            Plot.plot_legends(ax, 5, outfilename=plot_dirname+name+"_legends")
            
        elif (lockdown_num == 9):
            if (CITY != 'nyc'):
                print ("Real curve match is only for NYC now.")
                exit(0)	
            
            
            ## Match with the real curve.
            ## NYC data: https://raw.githubusercontent.com/nychealth/coronavirus-data/master/tests.csv
            fig, ax = plt.subplots()
            df1 = pd.read_csv('../data/tests.csv')
            
            
            label_config = LabelConfig(-1, prefix="Mobility simulation") #LabelConfig(R0_index, prefix="$R_0$=")
            p = Plot(result_dirname, configs["LOCKDOWN_9"], label_config, start_time, nusers)
            p.plot_growth_factor(ax=ax, xmin=0, ymin=0, xmax=114)
            ax.legend()
            
            ax.axhline(y=1, linestyle='--', color='k')

            real_infects = ndimage.gaussian_filter1d( df1['POSITIVE_TESTS'].to_numpy(), 2)
            growth_factor = Plot.get_growth_factor(real_infects)
            ax.plot(growth_factor, label="Real")
            
            #exit(0)
            '''
            fig, ax = plt.subplots()
            doubling_times = Plot.get_doubling_time(np.cumsum(real_infects))
            ax.plot(range(len(doubling_times)), doubling_times, label="Real")
            label_config = LabelConfig(R0_index, prefix="$R_0$=")
            p = Plot(result_dirname, configs["LOCKDOWN_0"], label_config, start_time, nusers)
            p.plot_doubling_times(ax=ax)
            '''
            #total_tests  = sum(df1['TOTAL_TESTS'].to_numpy())
            #cum_infects = np.cumsum(real_infects)/total_tests
            #ax.plot(range(len(cum_infects)), 100*cum_infects, label="Real infects")
            
            ax.legend()
            Plot.saveplotfile(fig, plot_dirname+"compare_real")
            
        elif(lockdown_num == 51): ## health value vs social value for different lockdown strategies.
            # Compare mitigation strategies
            compare_configs = []
            lockdown_desc = {"LOCKDOWN_1": "Staying home", \
                             "LOCKDOWN_2": "Close popular venues", \
                             "LOCKDOWN_3": "Protect active people", \
                             "LOCKDOWN_4": "Cleaning venue", \
                             "LOCKDOWN_5": "Group people",\
                             "LOCKDOWN_6": "Close random venues",\
                             "LOCKDOWN_7": "Protect random people",\
                             }
            for name in [ "LOCKDOWN_1", "LOCKDOWN_2", "LOCKDOWN_6", "LOCKDOWN_7", "LOCKDOWN_3"]:
                _cfg = configs[name]
                compare_configs.append( [lockdown_desc[name], _cfg] )
            #print (compare_configs)
            p = Plot(result_dirname, start_time = start_time, nusers=nusers)
            fig, ax = plt.subplots()
            nolockdown_config= [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]
            ax = p.compare_total_infect_vs_social_value(nolockdown_config, compare_configs, df, ax=ax)
            Plot.saveplotfile(fig, plot_dirname+"compare_strategies")
            Plot.plot_legends(ax, 5, outfilename=plot_dirname+"compare_strategies_legends")
            
        elif (lockdown_num == 61):
            p = Plot(result_dirname, start_time = start_time, nusers=nusers)
            fig, ax = plt.subplots(figsize=(8,6))
            nolockdown_config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]
            ax = p.tracking_popular_users(nolockdown_config, df, ax=ax, xmin=0, ymin=0, xmax=90)
            Plot.saveplotfile(fig, plot_dirname+"tracking_popular_users")
            Plot.plot_legends(ax, 3, outfilename=plot_dirname+"tracking_popular_users_legends")
        elif (lockdown_num == 62):
            p = Plot(result_dirname, start_time = start_time, nusers=nusers)
            fig, ax = plt.subplots(figsize=(8,6))
            nolockdown_config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]
            ax = p.tracking_popular_venues(nolockdown_config, df, ax=ax, xmin=0, ymin=0, xmax=90)
            Plot.saveplotfile(fig, plot_dirname+"tracking_popular_venues")
            Plot.plot_legends(ax, 1, outfilename=plot_dirname+"tracking_popular_venues_legends")

        '''
        elif (lockdown_num == 63):
            p = Plot(result_dirname, start_time = start_time, nusers=nusers)
            fig, ax = plt.subplots(figsize=(8,6))
            nolockdown_config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]
            ax = p.popular_venue_infection_vs_checkin(nolockdown_config, df, ax=ax)
            #Plot.saveplotfile(fig, plot_dirname+"popular_venue_infection_vs_checkin")
            #Plot.plot_legends(ax, 3, outfilename=plot_dirname+"popular_venue_infection_vs_checkin_legends")
        '''
        plt.show()
    else:
        ## Here add the strategies to execute.
        for name in [ "LOCKDOWN_0", "LOCKDOWN_100", "LOCKDOWN_1", "LOCKDOWN_2", "LOCKDOWN_3", "LOCKDOWN_4", "LOCKDOWN_5", "LOCKDOWN_6", "LOCKDOWN_7", "LOCKDOWN_81", "LOCKDOWN_82", "LOCKDOWN_83" ]:
        
            values = configs[name]
            print ("---------------------------name:", name, "----------------------------------")
            for _config in values:
                c = Configs(*_config)
                iteration = 10
                n_jobs = 10

                if (c._lockdown_num == 9):
                    restrictions = get_mobility_restrictions()
                    c._lockdown_arg = restrictions

                elif (c._lockdown_num == 11): ## Scaling experiment
                    __df = pd.read_csv("../data/"+CITY+"/checkin.csv")
                    for percentage in [25, 50, 75, 90, 100]:
                        c = Configs(*_config)
                        num_users = int(nusers * percentage /100.); print ("num_users:", num_users)
                        all_users = __df.groupby('userId').count()[['venueId']].index.tolist()
                        selected_users = np.random.choice(all_users, size=num_users, replace=False)
                        df = __df[__df['userId'].isin(selected_users)].copy().reset_index(drop=True)
                        
                        result = Parallel(n_jobs=10)(delayed(single_experiment)(df, start_time, nusers, nvenues, ncheckins, c) for _ in range(10))

                        _myconfig = copy.deepcopy(_config); _myconfig[lockdown_arg_index] = percentage
                        filename = "_".join([str(x) for x in _myconfig])
                        print("saving ", filename)
                        np.save(result_dirname+filename, result)
                elif (c._lockdown_num in [15, 16] ):
                    for percentage in [25, 50, 75, 100]:
                        c = Configs(*_config)
                        if (percentage == 100):
                            _df = df.copy()
                        else:
                            num_users = int(nusers * percentage /100.);
                            all_users = df.groupby('userId').count()[['venueId']].index.tolist()
                            selected_users = np.random.choice(all_users, size=num_users, replace=False)
                            _df = df[df['userId'].isin(selected_users)].copy().reset_index(drop=True)
                        
                        _nusers = len(_df.userId.unique())
                        _nvenues = len(_df.venueId.unique())
                        if (c._lockdown_arg == 0):
                            blacklist_users = []
                            blacklist_venues = []
                            __df = _df.copy()
                        else:
                            if (c._lockdown_num == 15):
                                rank = int(math.ceil(_nvenues*c._lockdown_arg/100.))
                                xf = _df.groupby('venueId').count()[['userId']]
                                sorted_xf = xf.sort_values('userId', ascending=False)
                                blacklist_venues = sorted_xf.index.tolist()[:rank]                      
                                __df = _df[~_df['venueId'].isin(blacklist_venues)].copy().reset_index(drop=True)
                            elif (c._lockdown_num == 16):
                                rank = int(math.ceil(_nusers*c._lockdown_arg/100.))
                                xf = _df.groupby('userId').count()[['venueId']]
                                sxf = xf.sort_values('venueId', ascending=False)
                                sxf.index.names = ['userId']
                                blacklist_users = sxf[:rank].index.to_list()
                                __df = _df[~_df['userId'].isin(blacklist_users)].copy().reset_index(drop=True)
                        
                        __nusers = len(__df.userId.unique())
                        __nvenues = len(__df.venueId.unique())
                        print ("++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        print ("percentage:", percentage, "arg:", c._lockdown_arg, "len(black_usrs):", len(blacklist_users),\
                                 "len(_df):", len(_df), len(__df), "nvenues:", _nvenues, __nvenues, "users:", _nusers, __nusers)
                        
                        
                        c._lockdown_num = 0 # Close popular venues are already deleted. So, no lockdown strategy needed.
                        
                        result = Parallel(n_jobs=n_jobs)(delayed(single_experiment)\
                                   (__df.copy(), start_time, __nusers, __nvenues, len(__df), c) for _ in range(iteration))
                        filename = "_".join([str(x) for x in _config])+"_"+str(percentage)
                        print("saving ", filename)
                        np.save(result_dirname+filename, result)

                        del _df
                        del __df
                        gc.collect()

                else:
                    
                    filename = "_".join([str(x) for x in _config])
                    print ("starting:", c, "filename:", filename)
                    c._city = CITY
                    #single_experiment(df, start_time, nusers, nvenues, ncheckins, c) ; exit(0)

                    print ("n_jobs:", n_jobs)
                    result = Parallel(n_jobs=n_jobs)(delayed(single_experiment)(df.copy(), start_time, nusers, nvenues, ncheckins, c) for _ in range(iteration))
                    

                    print("saving ", filename)
                    np.save(result_dirname+filename, result)
            print ("done.")



