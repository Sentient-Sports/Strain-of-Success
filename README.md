# The Strain of Success: A Predictive Model for Injury Risk Mitigation and Team Success in Soccer

Player injuries in soccer can significantly impact long-term team performance both on and off the pitch. According to insurance brokers Howden, throughout the ‘Big Five’ leagues in the 2021/22 season, injury costs for clubs reached £513 million, a 29% increase on the previous season.  Despite this, we find that managers still tend to focus on short-term gains with "greedy" strategies, often playing star players when they are at high risk of injury. Therefore, we have developed a data-driven team selection strategy that considers player injury risk and match outcome probabilities, reducing player injuries while sustaining long-term team success. Using this strategy, we found that our strategy achieved similar season expected points to a myopic greedy strategy (which we find to closely match real-world manager strategies) whilst reducing player injury by 5%, with an injury reduction of 13% for first-team players.

The use of our MCTS-based team selection strategy for reducing player injury risk over a season, whilst maintaining team performance, compared to real-world selections is visualised below.

<img src="https://github.com/GregSoton/SoccerTeamSelection/assets/96203800/7d4c97ce-874e-4eb7-863c-e8928cc2bc47" width=100% height=100%>

## Model Framework
The framework of our team selection model is visualised in the diagram below.

![image](https://github.com/GregSoton/SoccerTeamSelection/assets/96203800/768d1a14-fe72-4107-ae86-0b7424544f12)

## Directory Structure

- **data_collection:** These notebooks are used to convert the data from multiple sources (i.e., event data, injury data, weather data) and convert these into features such as the VAEP features for Section 5 and the injury features for the injury model in Section 4 of the paper.

- **experiments:** These notebooks are using to collect the data and run the experiments shown in Section 7 of the paper. Some of this data is extracted from runs of the team selection model on a compute cluster. If you would like to run these experiments on your own data, you must store the outputs of the complete MCTS simulation and extract the information that is used in the notebooks.

- **injury_model:**

- **match_predictions:**

- **team_selection:**

## Installing Requirements

## Code Workflow

## Data and References
