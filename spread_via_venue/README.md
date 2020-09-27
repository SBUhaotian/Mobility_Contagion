# Infection transmission via venue using Foursquare checkin datasets

The project uses Python 3. Required packages to run: numpy, scipy, pandas, matplotlib, networkx, and joblib.

The original datasets are downloaded from here: [https://drive.google.com/file/d/1PNk3zY8NjLcDiAbzjABzY5FiPAFHq6T8/view?usp=sharing]

To replicate the results the repo contains the preprocessed data for two cities in the data/ directory. For other cities please download the dataset and preprocess using the script in the preprocess directory as follows.

* Adjust the location of the `raw_Checkins_anonymized.txt` and `raw_POIs.txt` in the preprocess/preprocess notebook.
* Sequentially run the two code snippets in the notebook; You need to adjust the city name and the geographical bounding box to select the POIs inside the box.
* The script shall create a csv file in the preprocess directory. You need to copy the file in the data/ folder renaming the file as checkin.csv inside a folder named as the city name. Please follow the structures for the other cities provided.

## Mobility simulation

To execute the mobility simulatory run `run_mobility_simulator.py` using `python3 run_mobility_simulator <city name> run`. It'll pick the checkin.csv under that city directory. This shall simulate the scenarios mentioned in Line 488 in run_mobility_simulator.py and the parameter settings are mentioned in the same file. Each scenario corresponds to a name with the format 'LOCKDOWN_X' with 'X' being the number that denotes the scenario. This would run multiple iterations of the scenarios and use parallel processing to speed up things. This should create intermediate result files in the result directory.

To plot the results you can use the same run_mobility_simulator.py file with `python3 run_mobility_simulator.py <city name> plot <LOCKDOWN number>`.

## Graph simulation

Build the bipartite graphs using the notebook `preprocess/build_bipartite_graph`. Change the city name as required. Run the `user-venue-user_graph_simulator.py` for this purpose. Change the name of the city to pick the relevant checkin file.

Dataset characteristics are plotted from the `more_plots.py` file.
