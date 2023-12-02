This folder contains the initial injury and game data to build the required datasets to run the MDP. The user will need to access event data (free data available at: https://statsbomb.com/what-we-do/hub/free-data/) to be able to completely run the notebooks. 

We explain below the final set of directories expected in this folder after running code and creating new directories when required:

- **events**: This data is provided by StatsBomb. See the link to their free data above.
- **matches**: This data is provided by StatsBomb. See the link to their free data above.
- **lineups**: This data is provided by StatsBomb. See the link to their free data above.
- **game_data**: This contains lineup data with in-game injuries. These files can be computed in the DataCollection notebook.
- **team_data**: This contains injury features for specific teams. These files can be computed in the DataCollection notebook.
- **18-19_Squad**: This contains the available squad for each team in the 2018-19 EPL season. This is used to form the team in the MDP. These squads can be generated in the Team_Rewards notebook.
- **injury_data**: This contains our injury data extracted from tranfermrket.
- **greedy_dfs**: These contain the results of greedy season simulations for each team. These can be computed in the MDP notebook.
- **MCTSResults**: These contain the results of MCTS season simulations for each team. These can be computed in the MDP notebook. For our simulations, we ran our code on a compute cluster and added log files to this directory from the cluster.
- **overview_data**: Contains the games, players and teams for the EPL seasons. This is extracted from event data.
- **player_data**: This contains injury data for specific players. These files can be computed in the DataCollection notebook.
- **predictions**: Injury predictions are stored in this folder. These can be computed in the XGBoost_Injury_Model notebook.
- **Team_rewards_DF**: These are precomputed rewards for teams over the season. These are computed in the Team_Rewards notebook.
