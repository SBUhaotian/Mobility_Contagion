# Infection transmission via person using Tsinghua Wifi datasets and Zhengzhou electrical bike trajectory datasets.

The project uses Python 3. Required packages to run: numpy, scipy, pandas, matplotlib, networkx.

The original datasets are from:
Tsinghua Wifi dataset: Sui, K.et al.Characterizing and improving wifi latency in large-scale operational networks. InProceedings of the 14th324Annual International Conference on Mobile Systems, Applications, and Services, 347–360
Zhengzhou electrical bike dataset: Wang,  H. & Gao,  J.   Distributed human trajectory sensing and partial similarity queries.   In2020 19th ACM/IEEE327International Conference on Information Processing in Sensor Networks (IPSN), 253–264

The Tsinghua Wifi dataset and Zengzhou bike datasets cannot be made public due to our non-disclosure267agreement with the respective authorities to preserve privacy of the individuals.

The dataset stores the meetings between two users. The format of meetings is as follow:
meeting.csv: [user1, user2, beginning time, end time, meeting location]
checkin.gpickle: user: [[user, beginning time, end time, location], ...]

To replicate the results, please covert the checkin or trajectory dataset as the above format.
* Preprocess the datasets and create checkin.gpickle (checkin of each user every day) and meeting.csv (meeting events between two users)

## Mobility simulation
To execute the mobility simulatory run 'Mobility_contagion.py' using 'python3 Mobility_contagion.py'. it will read the corresponding files and out the total infection number, the number of active cases, the number of newly infections and R naught value every day in the separate .csv files.

To plot the results, you can read the corresponding .csv file and run 'plotter.py' with 'python3 plotter.py'.

## Graph simulation

To build the social graph, run 'Social_Contagion.py' using 'python3 Social_Contagion.py'. It will convert the dataset to social networks. The weight of edges represent the meeting frequency between two users.

The social graph can be implemented in contagion simulator automatically, changing the meeting .csv file to the edges and the infection probability is also changed.

The output is the same with the mobility simulation.
