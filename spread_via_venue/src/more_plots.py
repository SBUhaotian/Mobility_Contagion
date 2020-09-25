import pandas as pd
import numpy as np
import networkx as nx
import myutils
from mobility_simulator import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['legend.numpoints'] = 1
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18

from mobility_simulator import *

city_label = {"nyc":"New York City", "Tokyo":"Tokyo", "chicago":"Chicago", "istambul":"Istanbul", "los_angeles":"Los angeles", "jakarta":"Jakarta", "london":"London"}

def convert_log_ticks(ticks):
    ticklabels = []
    for l in ticks:
        ticklabels.append("$10^{"+str(int(l))+"}$")
    return ticklabels

def plot_histogram(arr, htype, nbins, alpha=1.0, label=None, ax=None):
    if (ax == None):
        fig, ax = plt.subplots()
    else:
        fig = None
    hist, bin_edges = np.histogram(arr, bins = nbins) ##max(max(arr), nbins))
    if (htype == "loglog"):
        ax.scatter(np.log(bin_edges[:-1]), np.log(hist), marker='o', alpha=alpha, s=20, label=label)
        ax.set_xticklabels(convert_log_ticks(ax.get_xticks())); ax.set_yticklabels(convert_log_ticks(ax.get_yticks()))
    elif (htype == "scatter"):
        ax.scatter(bin_edges[:-1], hist, marker='o', s=10, alpha=alpha, label=label)
    elif (htype == "line"):
        ax.plot(bin_edges[:-1], hist, alpha=alpha, label=label) #, marker='o', markersize=10)
    elif (htype == "bar"):
        ax.bar(bin_edges[:-1], hist, width = 100, alpha=alpha, label=label)

    return fig, ax

def plot_checkins():
    fig, ax = plt.subplots(figsize=(12, 3))
    for city in ["nyc", "Tokyo", "chicago", "istambul", "los_angeles", "jakarta", "london"]:
        df = pd.read_csv("../data/"+city+"/checkin.csv")
        start_time = df.iloc[0].posixtime
        arr = np.zeros(800)
        for t in df['posixtime'].tolist():
            d = int((t-start_time)/24.)
            arr[ d ] += 1
        arr = arr[:140]
        #plot_histogram(arr, "line", 50, ax=ax)
        ax.plot(range(len(arr)), arr, label=city_label[city])
    

    Plot.set_lims(ax, xmin=0, xmax=140, ymin=0, ymax=None)
    ticks = ax.get_yticks()
    #print (ticks)
    yticklabels = [str(int(t/1000))+'k' for t in ticks]
    yticklabels[0] = '0'
    #print (yticklabels)
    ax.set_yticklabels(yticklabels)
    
    ax.set_ylabel("# checkins")
    ax.set_xlabel("# days from start in $2012$")
    
    for sat in np.arange(0, 140, 7)+4:
        ax.axvspan(sat, sat+1, alpha=0.5, color='grey')
    months = ['Apr', 'May', 'Jun', 'Jul', 'Aug']
    months_duration = [30-3, 31, 30, 31, 31, 30]
    d = 0
    for m in [0, 1, 2, 3, 4]:
        d = d + months_duration[m]
        print (d)
        ax.axvline(d, color='k', ls='--')
    
    Plot.plot_legends(ax, 4, outfilename="../plot/temporal_checkins_legends")
    
# Datasets typically starts at Tue Apr 03 18:00:07 +0000 2012
def plot_checkin_weekly_all():
    fig, ax = plt.subplots(figsize=(12, 3))
    for city in ["nyc", "Tokyo", "chicago", "istambul", "los_angeles", "jakarta", "london"]:
        df = pd.read_csv("../data/"+city+"/checkin.csv")
        start_time = df.iloc[0].posixtime
        arr = np.zeros(168)
        zones={'nyc':-4, 'istambul':3, 'Tokyo':9, 'chicago':-5, 'los_angeles':-7, 'jakarta':7, 'london':0}
        for t in df['posixtime'].tolist():
            t =t+zones[city]
            t = t % 168
            d = int((t-start_time))
            arr[ d ] += 1
        arr = 100*arr / sum(arr)
        ax.plot(range(len(arr)), arr, label=city_label[city])
    
    #for d in np.arange(0, 170, 24):
    #    ax.axvspan(d, d+6, alpha=0.2)

    for d in np.arange(0, 170, 24) + 6:
        ax.axvline(d, color='k', ls='--')
    ax.set_xticks(np.arange(0, 170, 24) + 6 )
    
    ax.set_ylabel("% checkins")
    ax.set_xlabel("# hours ")
    Plot.set_lims(ax, xmin=0, xmax=168, ymin=0, ymax=None)
    
    Plot.plot_legends(ax, 4, outfilename="../plot/temporal_checkins_weekly_legends")
    Plot.saveplotfile(fig, "../plot/temporal_checkins_weekly_all")

def plot_checkin_weekly_two():
    fig, ax = plt.subplots(figsize=(12, 3))
    for city in ["nyc", "istambul"]:
        df = pd.read_csv("../data/"+city+"/checkin.csv")
        start_time = df.iloc[0].posixtime
        arr = np.zeros(168)
        zones={'nyc':-4, 'istambul':3, 'Tokyo':9, 'chicago':-5, 'los_angeles':-7, 'jakarta':7, 'london':0}
        for t in df['posixtime'].tolist():
            t =t+zones[city]
            t = t % 168
            d = int((t-start_time))
            arr[ d ] += 1
        arr = 100*arr / sum(arr)
        ax.plot(range(len(arr)), arr, label=city_label[city])
    
    for d in np.arange(0, 170, 24) + 6:
        ax.axvline(d, color='k', ls='--')
    ax.set_xticks(np.arange(0, 170, 24) + 6 )

    for d in np.arange(0, 170, 24):
        ax.axvspan(d, d+12, color='k', alpha=0.2) # 18:00 - 6:00
        ax.axvspan(d+12, d+24, color='y', alpha=0.2) #

    ax.set_xticks(np.arange(0, 170, 24) + 6 )
    
    ax.set_ylabel("% checkins")
    ax.set_xlabel("# hours ")
    Plot.set_lims(ax, xmin=0, xmax=168, ymin=0, ymax=None)
    #ax.legend()
    
    Plot.plot_legends(ax, 4, outfilename="../plot/temporal_checkins_weekly_two")
    Plot.saveplotfile(fig, "../plot/temporal_checkins_weekly_two")


def plot_user_checkins_histogram():
    fig, ax = plt.subplots()
    for city in ["nyc", "Tokyo", "chicago", "istambul", "los_angeles", "jakarta", "london"]:
        df = pd.read_csv("../data/"+city+"/checkin.csv")
        num_checkins = len(df)
        xf = df.groupby('userId').count()['venueId'].to_numpy() #/ float(num_checkins)
        
        plot_histogram(xf, "line", 100, ax=ax, label=city)

    #xmin, xmax = ax.get_xlim(); ax.set_xlim([1, xmax])
    ax.set_yscale('log'); ax.set_xscale('log')
    #ax.legend()
    ax.set_xlabel("# checkins by a person (log scale)")
    ax.set_ylabel("# such perople (log scale)")
    Plot.saveplotfile(fig, "../plot/user_histogram")


def plot_user_checkins_histogram_nyc():
    from sklearn import linear_model
    
    fig, ax = plt.subplots()
    city = 'Tokyo'
    df = pd.read_csv("../data/"+city+"/checkin.csv")
    num_checkins = len(df)
    xf = np.sort(df.groupby('userId').count()['venueId'].to_numpy()) #/ float(num_checkins)
    
    def _f(arr, bins):
        hist, bin_edges = np.histogram(arr, bins = bins)
        y = hist[np.where(hist>0)[0]]
        x = bin_edges[:-1][np.where(hist>0)[0]]

        x = np.log10(x)
        y = np.log10(y)
        ax.scatter(x, y) #, label=label)

    index = max(np.where(xf<100)[0])
    print ("index:", index)
    low_active_arr = xf[ :index ]
    high_active_arr = xf[index: ]
    print ("num users with < 100 checkins:", len(low_active_arr), "num checkins from users with < 100 checkins:", np.sum(low_active_arr))
    print ("num users with >= 100 checkins:", len(high_active_arr), "num checkins from users with >= 100 checkins:", np.sum(high_active_arr) )
    print ("num users:", len(xf), "num checkins:", sum(xf))

    _f(xf, 100)

    xmin, xmax = ax.get_xlim()

    ax.set_xticklabels(convert_log_ticks(ax.get_xticks())); 
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(convert_log_ticks(ax.get_yticks()))
    Plot.saveplotfile(fig, "../plot/user_histogram_"+city)

def plot_venue_checkins_histogram():
    fig, ax = plt.subplots()
    for city in ["nyc", "Tokyo", "chicago", "istambul", "los_angeles", "jakarta", "london"]:
        df = pd.read_csv("../data/"+city+"/checkin.csv")
        num_checkins = len(df)
        xf = 100*df.groupby('venueId').count()['userId'].to_numpy()#/ float(num_checkins)
        plot_histogram(xf, "line", 200, ax=ax, alpha=1.0, label=city)

    ax.set_yscale('log'); ax.set_xscale('log')
    ax.set_xlabel("# checkins at a venue (log scale)")
    ax.set_ylabel("# such venues (log scale)")
    Plot.saveplotfile(fig, "../plot/venue_histogram")




def plot_venue_checkins_histogram_nyc():
    fig, ax = plt.subplots()
    city = 'Tokyo'
    df = pd.read_csv("../data/"+city+"/checkin.csv")
    num_checkins = len(df)
    xf = np.sort(df.groupby('venueId').count()['userId'].to_numpy()) #/ float(num_checkins)
    

    num_checkins = sum(xf)
    num_venues = len(xf)
    index = min(np.where( np.cumsum(xf) >= sum(xf)/2. )[0])
    print ("venues with 50:", (num_venues-index)*100./num_venues, (num_checkins-sum(xf[:index]))*100./num_checkins)
    
    low_active_arr = xf[ :index ]
    high_active_arr = xf[index: ]
    def _f(arr, bins):
        hist, bin_edges = np.histogram(arr, bins = bins)
        y = hist[np.where(hist>0)[0]]
        x = bin_edges[:-1][np.where(hist>0)[0]]

        x = np.log10(x)
        y = np.log10(y)
        ax.scatter(x, y, color='#1f77b4', s=50)

    _f(xf, 100)

    ax.set_xticks([0, 2, 4])
    ax.set_xticklabels(convert_log_ticks(ax.get_xticks())); 
    ax.set_yticks([0, 2, 5])
    ax.set_yticklabels(convert_log_ticks(ax.get_yticks())); 
    
    Plot.saveplotfile(fig, "../plot/venue_histogram_"+city)


def venue_infection_histogram():
    fig, ax = plt.subplots()
    config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]
    filename = "_".join([str(x) for x in config])
    
    for city in ["nyc", "Tokyo", "chicago", "istambul", "los_angeles", "jakarta", "london"]:
        results = np.load("../result/mobility/"+city+"/"+filename+".npy", allow_pickle=True)
        df = pd.read_csv("../data/"+city+"/checkin.csv")
        venues = df.venueId.unique().tolist()
        venue_dict = {k: np.zeros(10) for k, v in zip(venues, np.zeros(len(venues), dtype='int'))}
        print ("number of venues:", len(venue_dict))
        
        it_index = 0
        for user_dict, _, _ in results:
            for uid in user_dict.keys():
                if (user_dict[uid].expose_time > 0 and user_dict[uid].infect_from_venue != None):
                    venue_dict[user_dict[uid].infect_from_venue][it_index] += 1
            it_index += 1
        
        arr = np.array([])
        for k, v in venue_dict.items():
            arr = np.append (arr, np.median (v))
        arr = np.sort(arr)
        hist, bin_edges = np.histogram(arr, bins = 100)
        y = hist[np.where(hist>0)[0]]
        x = bin_edges[:-1][np.where(hist>0)[0]]
        ax.plot(x, y, label=city_label[city])
    ax.set_xscale('symlog'); ax.set_yscale('symlog')
    ax.set_xlabel("# of people infected from a venue (log)")
    ax.set_ylabel("# of such venues (log)")

    Plot.saveplotfile(fig, "../plot/venue_infection_histogram")
    
    ax.legend()

def venue_infection_histogram_nyc():
    fig, ax = plt.subplots()
    config = [10, 0.75, 24*2, 0.35, 0, 0, 0, 0]
    filename = "_".join([str(x) for x in config])
    
    for city in ["nyc"]:
        results = np.load("../result/mobility/"+city+"/"+filename+".npy", allow_pickle=True)
        df = pd.read_csv("../data/"+city+"/checkin.csv")
        venues = df.venueId.unique().tolist()
        venue_dict = {k: np.zeros(10) for k, v in zip(venues, np.zeros(len(venues), dtype='int'))}
        print ("number of venues:", len(venue_dict))
        
        it_index = 0
        for user_dict, _, _ in results:
            for uid in user_dict.keys():
                if (user_dict[uid].expose_time > 0 and user_dict[uid].infect_from_venue != None):
                    venue_dict[user_dict[uid].infect_from_venue][it_index] += 1
            it_index += 1
        
        arr = np.array([])
        for k, v in venue_dict.items():
            #print (v)
            arr = np.append (arr, np.median (v))
        arr = np.sort(arr)
        print ("arr:", arr)
        # We want statistics like this: 50% people are infected from 0.05% most popular venues.
        hist, bin_edges = np.histogram(arr, bins = 100)
        print (np.cumsum(arr))
        print(len(np.where(np.cumsum(arr) >= (sum(arr) / 2.))[0]))
        exit(0)
        y = hist[np.where(hist>0)[0]]
        x = bin_edges[:-1][np.where(hist>0)[0]]
        ax.scatter(x, y)
    ax.set_xscale('symlog'); ax.set_yscale('symlog')
    ax.set_xlabel("# of people infected from a venue (log)")
    ax.set_ylabel("# of such venues (log)")

    Plot.saveplotfile(fig, "../plot/venue_infection_histogram_nyc")
    
    #ax.legend()


if __name__ == "__main__":
    #plot_venue_checkins_histogram()
    #plot_contact_graph()
    #plot_checkins()
    #plot_checkin_weekly_all()
    #plot_checkin_weekly_two()
    
    
    #plot_user_checkins_histogram()
    #plot_user_checkins_histogram_nyc()
    #plot_venue_checkins_histogram_nyc()
    
    #venue_infection_histogram()
    #venue_infection_histogram_nyc()
    
    #plot_graph_components_vs_threshold()
    #plot_contact_graph()
    #test()
    #plot_graph_comps_vs_popular_venue_isolating()
    #plot_contact_graph()
    
    plt.show()
    
