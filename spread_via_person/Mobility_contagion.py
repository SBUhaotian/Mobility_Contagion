import random
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter

import csv
import time
import os

INCUBATION_TIME = 6*24
STD_INCUBATION_TIME = 1*24
SYMPTOM_REMOVAL_TIME = 5*24
ASYMPTOM_REMOVAL_tIME = 19*24
STD_REMOVAL_TIME = 1*24

ASYMPTOM_RATE = 0.35
SIMULATION_DAY = 120

LOCKDOWN_0 = 0 # No Mitigation Strategy
LOCKDOWN_1 = 1 # Reduce Checkin Number
LOCKDOWN_2 = 2 # Isolate the popular people
LOCKDOWN_3 = 3 # Close the popular venues
LOCKDOWN_4 = 4 # Different user groups operate in cycle

LOCKDOWN_STRATEGY = [LOCKDOWN_0, LOCKDOWN_1, LOCKDOWN_2, LOCKDOWN_3, LOCKDOWN_4]
# Skip probabilities
lockdown_1_config = [ 0.1, 0.2, 0.4 ]
# Percentile of popular venues to be completely locked down
lockdown_2_config = [ 0.05, 0.1, 0.15 ]
# Percentile of popular users to be completely isolated
lockdown_3_config = [ 0.1, 0.2, 0.4 ]
# Build x number of random user groups. Allow go out rule in round robin
lockdown_4_config = [ 1, 2, 4, 8]

lockdown_config = {\
            LOCKDOWN_0: [None], \
            LOCKDOWN_1: lockdown_1_config,\
            LOCKDOWN_2: lockdown_2_config,\
            LOCKDOWN_3: lockdown_3_config,\
            LOCKDOWN_4: lockdown_4_config\
            }

 
def infection_process(user, seed_num, random_user, initial_R, strategy, lockdown_parameter, file_para):
    user_list = list(user.keys())
    # initial_R = infection probability * average number of meetings * contagion days
    inf_prob = initial_R/(90143.6/len(user_list))/9.55

    res_inf_total = []
    res_R_value = []
    res_active = []

    for iteration in range(25):
        # Incubation days of each user
        incubation = user.copy()
        # Removal days of each user
        removal = user.copy()
        # Get infected from whom
        get_infected = user.copy()

        if (strategy == LOCKDOWN_4):
            # Group Index for each user
            group = user.copy()
            for u in user:
                group[u] = random.randint(0, lockdown_parameter - 1)
        
        for i in range(seed_num):
            # Infection time of this user
            user[random_user[iteration][i]] = 0.1
            incubation[random_user[iteration][i]] = 0
            get_infected[random_user[iteration][i]] = 'seed'
            if (random.random() < ASYMPTOM_RATE):
                removal[random_user[iteration][i]] = np.random.normal(ASYMPTOM_REMOVAL_tIME, STD_REMOVAL_TIME, 1)[0]
            else:
                removal[random_user[iteration][i]] = np.random.normal(SYMPTOM_REMOVAL_TIME, STD_REMOVAL_TIME, 1)[0]
        
        temp_res = [10]
        for day in range(SIMULATION_DAY):
            file_name = "event"+str(day)+".csv"
            with open(file_name, "r") as f:
                reader = csv.reader( (line.replace('\0','') for line in f) )
                    next(reader, None)
                    for row in reader:
                        u1 = row[1]
                        u2 = row[2]
                        loc = row[5]

                        # Two users have already been infected
                        if (user[u1] > 0) and (user[u2]>0):
                            continue
                        # Two users are healthy now
                        if (user[u1] == 0) and (user[u2] == 0):
                            continue
                        
                        if (random.random() > inf_prob):
                            continue
                        if (strategy == LOCKDOWN_1):
                            # One of checkins is skipped
                            if (random.random() > lockdown_parameter):
                                continue
                        if (strategy == LOCKDOWN_2):
                            # One of user is isolated
                            if (u1 in lockdown_parameter) or (u2 in lockdown_parameter):
                                continue
                        if (strategy == LOCKDOWN_3):
                            # The meeting venue is closed
                            if (loc in lockdown_parameter):
                                continue
                        if (strategy == LOCKDOWN_4):
                            # Two users are not in the same group
                            if (group[u1] != group[u2]):
                                continue
                        
                        if (user[u2] > 0):
                            temp_r = u2
                            u2 = u1
                            u1 = temp_r
                        
                        if (user[u1] + removal[u1] + incubation[u1] > day*24 + int(row[3])/3600) and (user[u1] + incubation[u1] <= (day*24 + int(row[3])/3600) + 0.1):
                            get_infected[u2] = u1
                            user[u2] = (day*24 + int(row[3])/3600) + 1

                            incubation[u2] = np.random.normal(INCUBATION_TIME, STD_INCUBATION_TIME,1)[0]
                            if (random.random() < ASYMPTOM_RATE):
                                removal[u2] = np.random.normal(ASYMPTOM_REMOVAL_tIME, STD_REMOVAL_TIME, 1)[0]
                            else:
                                removal[u2] = np.random.normal(ASYMPTOM_REMOVAL_tIME, STD_REMOVAL_TIME, 1)[0]

                    infected_num = 0
                    for u in user_list:
                        if (user[u] > 0):
                            infected_num += 1
                    temp_res.append(infected_num)
        
        active_case = np.zeros(SIMULATION_DAY+50)
        infected_from = np.zeros(SIMULATION_DAY+50)

        for u in user_list:
            if (user[0] == 0):
                continue
            for i in range(int(user[u]/24), int((user[u] + removal[u] + incubation[u])/24)+1):
                active_case[i] += 1
        print("Active Cases: ", active_case)

        for u in user_list:
            if (get_infected[u] == 0) or (get_infected[u] == 'seed'):
                continue
            if (get_infected[u] == 0.1):
                infected_from[0] += 1
            else:
                infected_from[int((user[get_infected[u]]-1)/24)+1] += 1
        
        Rvalue = []
        for i in range(SIMULATION_DAY):
            if (i == 0):
                Rvalue.append(infected_from[i]/10)
            else:
                if (temp_res[i] - temp_res[i-1] == 0):
                    Rvalue.append(0)
                else:
                    Rvalue.append(infected_from[i] / (temp_res[i] - temp_res[i-1]))
        print("R value: ", Rvalue)
        print("Total Infection Number: ", temp_res)
        res_inf_total.append(temp_res)
        res_R_value.append(Rvalue)
        res_active.append(active_case)
    
    total_df = pd.DataFrame(data = res)
    total_df.to_csv("Total_Infection_"+strategy+"_"+str(file_para)+".csv")
    active_df = pd.DataFrame(data = res_active)
    active_df.to_csv("Active_Case_"+strategy+"_"+str(file_para)+".csv")
    Rnaught_df = pd.DataFrame(data = res_R)
    Rnaught_df.to_csv("Rnaught_"+strategy+"_"+str(file_para)+".csv")

def get_popular_user(parameter):
    user = nx.read_gpickle("user.gpickle")
    for day in range(30):
        checkin = nx.read_gpickle("Checkin"+str(day)+".gpickle")
        for u in checkin.keys():
            user[u] += len(checkin[u])
    checkin_num = []
    for u in user.keys():
        checkin_num.append([u, user[u]])
    checkin_num = sorted(checkin_num, key=lambda x:x[1], reverse = True)
    popular_list = []
    for i in range(int(parameter * len(checkin_num))):
        popular_list.append(checkin_num[i][0])
    return popular_list

def get_popular_venue(parameter):
    loc = dict()
    for day in range(30):
        checkin = nx.read_gpickle("Checkin"+str(day)+",gpickle")
        for u in checkin.keys():
            for visit in checkin[u]:
                if (visit[3] not in loc.keys()):
                    loc[visit[3]] = 0
                loc[visit[3]] += 1
    venue_num = []
    for v in loc.keys():
        venue_num.append([v, loc[v]])
    venue_num = sorted(venue, key=lambda x:x[1], reverse = True)
    popular_list = []
    for i in range(int(parameter * len(venue_num))):
        popular_list.append(venue[i][0])
    return popular_list


if __name__ == "__main__":
    user = nx.read_gpickle("users.gpickle")
    num_seed = 10
    initial_R = 3

    random_user = []
    with open("Random_list.csv", "r") as f:
        reader = csv.reader( (line.replace('\0','') for line in f) )
        next(reader, None)
        for row in reader:
            random_user.append(row[1:])

    for i in range(5):
        strategy = LOCKDOWN_STRATEGY[i]
        config = lockdown_config[strategy]
        for j in range(len(config)):
            if (strategy == LOCKDOWN_2):
                lockdown_parameter = get_popular_user(config[j])
            if (strategy == LOCKDOWN_3):
                lockdown_parameter = get_popular_venue(config[j])
            infection_process(user, seed_num, random_user, initial_R, strategy, lockdown_parameter, config[j])

                    