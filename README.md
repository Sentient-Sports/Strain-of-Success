# The Strain of Success: A Predictive Model for Injury Risk Mitigation and Team Success in Soccer

Player injuries in soccer can significantly impact long-term team performance both on and off the pitch. According to insurance brokers Howden, throughout the ‘Big Five’ leagues in the 2021/22 season, injury costs for clubs reached £513 million, a 29% increase on the previous season.  Despite this, we find that managers still tend to focus on short-term gains with "greedy" strategies, often playing star players when they are at high risk of injury. Therefore, we have developed a data-driven team selection strategy that considers player injury risk and match outcome probabilities, reducing player injuries while sustaining long-term team success. Using this strategy, we found that our strategy achieved similar season expected points to a myopic greedy strategy (which we find to closely match real-world manager strategies) whilst reducing player injury by 5%, with an injury reduction of 13% for first-team players. The paper for this work will be released after Sloan Sports submission review.

The use of our MCTS-based team selection strategy for reducing player injury risk over a season, whilst maintaining team performance, compared to real-world selections can be visualised in an example below (where dots represent rests).

<img src="https://github.com/GregSoton/SoccerTeamSelection/assets/96203800/7d4c97ce-874e-4eb7-863c-e8928cc2bc47" width=100% height=100%>

## Model Framework
The framework of our team selection model is visualised in the diagram below.

![image](https://github.com/GregSoton/SoccerTeamSelection/assets/96203800/768d1a14-fe72-4107-ae86-0b7424544f12)

## Directory Structure

- **data_collection:** These notebooks are used to collate the data from multiple sources (i.e., event data, injury data, weather data) and convert these into features such as the VAEP features for Section 5 and the injury features for the injury model in Section 4 of the paper.

- **experiments:** These notebooks are using to collect the data and run the experiments shown in Section 7 of the paper. Some of this data is extracted from runs of the team selection model on a compute cluster. If you would like to run these experiments on your own data, you must store the outputs of the complete MCTS simulation and extract the information that is used in the notebooks.

- **injury_model:** This directory contains the code for training and saving the injury model presented in Section 4 of the paper.

- **match_predictions:** This directory contains the code to train and save the match outcome prediction model presented in Section 5 of the paper.

- **team_selection_MDP:** This directory contains the code to generate the team selection MDP and find optimal solutions using the MCTS algorithm.

## Installing Requirements

Firstly, ensure that the requirement.txt file is in the directory so that the requirements can be installed to run the code in this repository. Next, run the following command in the command prompt for the directory:

```
pip install -r requirements.txt
```

## Code Workflow

- Run the notebooks in the data_collection directory to obtain the injury and VAEP feature sets.

- Run the XGBoost_Injury_Model notebook in the injury_model directory to train the injury prediction model. This model can be saved as a pickle file and used in the team selection MDP.

- Run the Team_Rewards notebook in the match_predictions directory to train the match outcome prediction model (reward function). The rewards for teams at each game are precomputed and stored in this notebook to improve the computational efficiency of the team selection MDP and MCTS algorithm.
  
- Run the MDP notebook in the team_selection_MDP directory to run the Markov Decision Process model of team selection over a season. This notebook also contains the code to run the MCTS algorithm over a season to get a complete season simulation. To get an accurate measure of the success of the MCTS algorithm, this process must be repeated and data logged in files to retrieve many season simulations for the algorithm. Furthermore, the notebook (aswell as the updated_injury_probs.py and the MDP.py files) contain team ID variables. These variables must be updated depending on the team you would like to run the algorithm for.

- The MCTSResultsAnalysis notebook in the experiments directory can be used to find the overall results of many MCTS season simulations so that these results can be compared to the greedy strategy. This notebook, aswell as the MDP notebook, also contains some code that is used to store data ready to be used for the experiments

- The ExperimentPlots notebook in the experiments directory can be used to display all the plots that were used in the paper. These use results collected from the other notebooks and stored in spreadsheets when running MCTS on a compute cluster. 

## Data and References

- Injury Data: The player injury data was provided from Transfermrket - https://www.transfermarkt.co.uk/.

- Events Data: For this work, events and lineup data is used for the English Premier League 2017/18 and 2018/19 seasons. This data has been provided by StatsBomb. Free event and lineup data samples can be accessed from StatsBomb: https://statsbomb.com/what-we-do/hub/free-data/.

- Wage Data: We accessed the wage data for the 2018/19 English Premier League season from Capology: https://www.capology.com/uk/premier-league/salaries/2018-2019/.

- The original VAEP model can be viewed in the following repository: https://github.com/ML-KULeuven/socceraction.
