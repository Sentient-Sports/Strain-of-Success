from pandera.typing import DataFrame
from socceraction.data.statsbomb.schema import StatsBombPlayerSchema
import socceraction.data.statsbomb.loader as sch
from typing import cast
from socceraction.spadl.schema import SPADLSchema
import socceraction.spadl.statsbomb as sb
import socceraction.spadl.base as base
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action="ignore", message="credentials were not supplied. open data access only")
import tqdm
from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl as spadl

def players_func(self, game_id: int) -> DataFrame[StatsBombPlayerSchema]:
        
        cols = [
            "game_id",
            "team_id",
            "player_id",
            "player_name",
            "is_starter",
            "starting_position_id",
            "starting_position_name",
            "minutes_played",
        ]

        obj = self._lineups(game_id)
        playersdf = pd.DataFrame(sch._flatten_id(p) for lineup in obj for p in lineup["lineup"])
        playergamesdf = sch.extract_player_games(self.events(game_id))
        playersdf = pd.merge(
            playersdf,
            playergamesdf[
                ["player_id", "team_id", "position_id", "position_name", "minutes_played"]
            ],
            on="player_id",
        )
        playersdf["game_id"] = game_id
        playersdf["position_name"] = playersdf["position_name"].replace(0, "Substitute")
        playersdf["position_id"] = playersdf["position_id"].fillna(0).astype(int)
        playersdf["is_starter"] = playersdf["position_id"] != 0
        playersdf.rename(
            columns={
                "player_nickname": "nickname",
                "country_name": "country",
                "position_id": "starting_position_id",
                "position_name": "starting_position_name",
            },
            inplace=True,
        )
        return cast(DataFrame[StatsBombPlayerSchema], playersdf[cols])
    
def convert_to_actions(events: pd.DataFrame, home_team_id: int) -> DataFrame[SPADLSchema]:
    """
    Convert StatsBomb events to SPADL actions.
    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing StatsBomb events from a single game.
    home_team_id : int
        ID of the home team in the corresponding game.
    Returns
    -------
    actions : pd.DataFrame
        DataFrame with corresponding SPADL actions.
    """
    actions = pd.DataFrame()

    events = events.copy()
    events['extra'].fillna({}, inplace=True)
    events.fillna(0, inplace=True)

    actions['game_id'] = events.game_id
    actions['original_event_id'] = events.event_id
    actions['period_id'] = events.period_id

    actions['time_seconds'] = (
        60 * events.minute
        + events.second
        - ((events.period_id > 1) * 45 * 60)
        - ((events.period_id > 2) * 45 * 60)
        - ((events.period_id > 3) * 15 * 60)
        - ((events.period_id > 4) * 15 * 60)
    )
    actions['team_id'] = events.team_id
    actions['player_id'] = events.player_id

    actions['start_x'] = events.location.apply(lambda x: x[0] if x else 1).clip(1, 120)
    actions['start_y'] = events.location.apply(lambda x: x[1] if x else 1).clip(1, 80)
    actions['start_x'] = ((actions['start_x'] - 1) / 119) * 105
    actions['start_y'] = 68 - ((actions['start_y'] - 1) / 79) * 68

    end_location = events[['location', 'extra']].apply(sb._get_end_location, axis=1)
    actions['end_x'] = end_location.apply(lambda x: x[0] if x else 1).clip(1, 120)
    actions['end_y'] = end_location.apply(lambda x: x[1] if x else 1).clip(1, 80)
    actions['end_x'] = ((actions['end_x'] - 1) / 119) * 105
    actions['end_y'] = 68 - ((actions['end_y'] - 1) / 79) * 68

    actions['type_name'] = events['type_name']
    #actions[['type_id', 'result_id', 'bodypart_id']] = events[['type_name', 'extra']].apply(
    #    _parse_event, axis=1, result_type='expand'
    #)

    actions.sort_values(['game_id', 'period_id', 'time_seconds']).reset_index(drop=True)
    
    actions = base._fix_direction_of_play(actions, home_team_id)

    actions['action_id'] = range(len(actions))
    #actions = _add_dribbles(actions)

    return cast(DataFrame[SPADLSchema], actions)

#If you want to go back to the days+1 route, simply change next date index to date + timedelta(days=1)
def get_injury_df_for_game(game_id, games_df, injury_df_formatted):
    lineup = pd.read_csv('game_data/'+str(game_id)+'/'+str(game_id)+'_lineup.csv')
    lineup_pids = lineup['player_id']
    date = games_df[games_df['game_id'] == game_id]['game_date'].values[0].split(' ')[0]
    date = datetime.strptime(date, '%Y-%m-%d')
    injured = []
    injury_types = []
    for l in lineup_pids:
        player_features = pd.read_csv('player_data/'+str(l)+'/'+str(l)+'_fixtures.csv')
        player_features_dates = pd.Series([datetime.strptime(d,'%Y-%m-%d') for d in player_features['game_date'].dropna()])
        #try:
        #    curr_date_index = player_features_dates[player_features_dates == date].index.values[0]
        #    next_date = player_features_dates[curr_date_index+1]
        #except:
        #    next_date = date + timedelta(days=7)
        next_date = date + timedelta(days=1)
        injury_dates = injury_df_formatted[injury_df_formatted['sb_id'] == l]
        count=0
        injury_type = None
        for i,row in injury_dates.iterrows():
            try:
                before_next = ((datetime.strptime(row['from'], '%Y-%m-%d') < next_date) & (datetime.strptime(row['from'], '%Y-%m-%d') < (date + timedelta(days=7))))
                if ((datetime.strptime(row['from'], '%Y-%m-%d')) >= date) & before_next:
                    count += 1
                    injury_type = injury_dates.loc[i,'Injury']
            except:
                print(row['from'])
        if count > 0:
            injured.append(True)
            injury_types.append(injury_type)
        else:
            injured.append(False)
            injury_types.append(injury_type)
    lineup['injured'] = injured
    lineup['injury_type'] = injury_types
    injured_game = lineup.copy()
    return injured_game

#If you want to go back to the days+1 route, simply change next date index to date + timedelta(days=1)
def get_injury_df_for_game_FFScout(game_id, games_df, injury_df_formatted):
    lineup = pd.read_csv('game_data/'+str(game_id)+'/'+str(game_id)+'_lineup.csv')
    lineup_pids = lineup['player_id']
    date = games_df[games_df['game_id'] == game_id]['game_date'].values[0].split(' ')[0]
    date = datetime.strptime(date, '%Y-%m-%d')
    injured = []
    injury_types = []
    for l in lineup_pids:
        #print("Player: ", l)
        player_features = pd.read_csv('player_data/'+str(l)+'/'+str(l)+'_fixtures.csv')
        player_features_dates = pd.Series([datetime.strptime(d,'%Y-%m-%d') for d in player_features['game_date'].dropna()])
        #print("Dates: ", player_features_dates.head(50))
        try:
            curr_date_index = player_features_dates[player_features_dates == date].index.values[0]
            next_date = player_features_dates[curr_date_index+1]
        except:
            next_date = date + timedelta(days=7)
        #print("Curr Date: ", date)
        #print("Next Date: ", min(next_date,date + timedelta(days=7))
        injury_dates = injury_df_formatted[injury_df_formatted['player_id'] == l]
        #print("Injury dates: ", injury_dates.head(50))
        count=0
        injury_type = None
        for i,row in injury_dates.iterrows():
            before_next = ((datetime.strptime(row['injury_date'], '%Y-%m-%d') < next_date) & (datetime.strptime(row['injury_date'], '%Y-%m-%d') < (date + timedelta(days=7))))
            if ((datetime.strptime(row['injury_date'], '%Y-%m-%d')) >= date) & before_next:
                count += 1
                injury_type = injury_dates.loc[i,'injury_type']
            #except:
            #    print(row['injury_date'])
        if count > 0:
            #print(True)
            injured.append(True)
            injury_types.append(injury_type)
        else:
            #print(False)
            injured.append(False)
            injury_types.append(injury_type)
    lineup['injured'] = injured
    lineup['injury_type'] = injury_types
    injured_game = lineup.copy()
    return injured_game