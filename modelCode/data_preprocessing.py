from __future__ import print_function

import logging
import modelCode
import pandas as pd
import json
from urllib.request import urlopen
from pathlib import Path
from datetime import timedelta

schema_file = Path('csv_files/final_schema.csv')
players_path22 = Path("csv_files/2022_playerstats_epl.csv")
players_path21 = Path("csv_files/2021_playerstats_epl.csv")
gk_path22 = Path("csv_files/2022_goalkeeperstats_epl.csv")
gk_path21 = Path("csv_files/2021_goalkeeperstats_epl.csv")
url_2021 = "https://fixturedownload.com/feed/json/epl-2021"


def unpack_json_data(url):
    invalid_teams = ["Newcastle", "Man Utd", "Man City", "Leicester", "Leeds", "Spurs", "Norwich"]
    valid_teams = ["Newcastle Utd", "Manchester Utd", "Manchester City", "Leicester City", "Leeds United", "Tottenham",
                   "Norwich City"]
    response = urlopen(url)
    data_json = json.loads(response.read())
    for dictionary in data_json:
        for key, value in dictionary.items():
            if value in invalid_teams:
                index = invalid_teams.index(value)
                dictionary[key] = valid_teams[index]
    return data_json

'''when called checks to see if dataframes are up to date, if player db out of date calls calls web scraper
if fixture db (final_schema) out of date calls create_training_schema()'''
def format_dataframes():
    if players_path22.is_file() and gk_path22.is_file():
        live_data = unpack_json_data(url_2021)
        live_data_df = pd.DataFrame(live_data)
        live_data_df.dropna(subset=['HomeTeamScore'], inplace=True)
        live_data_df["DateUtc"] = pd.to_datetime(live_data_df["DateUtc"], format = "%Y-%m-%d").dt.date
        live_data_df.sort_values(by="DateUtc", inplace=True)
        newest_live = live_data_df["DateUtc"].iloc[-1]
        newest_live = pd.to_datetime(newest_live, format='%Y-%m-%d').date()
        newest_live = newest_live - timedelta(days=7)
        df_player_stats22 = pd.read_csv("csv_files/2022_playerstats_epl.csv")
        sorted_df = df_player_stats22.sort_values(by="Date")
        newest_fbref = sorted_df["Date"].iloc[-1]
        newest_fbref = pd.to_datetime(newest_fbref, format='%Y-%m-%d').date()
        if newest_live > newest_fbref:
            # modelCode.scrapeStats("2022", "EPL")
            pass
    else:
        modelCode.scrapeStats("2022", "EPL")
    if players_path21.is_file() and gk_path21.is_file():
        df_player_stats21 = pd.read_csv("csv_files/2021_playerstats_epl.csv")
        sorted_df = df_player_stats21.sort_values(by="Date")
        newest_fbref21 = sorted_df["Date"].iloc[-1]
        if newest_fbref21 != "2021-05-23":
            modelCode.scrapeStats("2021", "EPL")
    else:
        modelCode.scrapeStats("2021", "EPL")

    player_path = Path("csv_files/2021_22_playerstats_epl.csv")
    gk_path = Path("csv_files/2021_22_goalkeeperstats_epl.csv")
    df_player_stats22 = pd.read_csv("csv_files/2022_playerstats_epl.csv")
    df_gk_stats22 = pd.read_csv("csv_files/2022_goalkeeperstats_epl.csv")
    df_player_stats21 = pd.read_csv("csv_files/2021_playerstats_epl.csv")
    df_gk_stats21 = pd.read_csv("csv_files/2021_goalkeeperstats_epl.csv")
    df_list = [df_player_stats22, df_player_stats21, df_gk_stats21, df_gk_stats22]
    for df in df_list:
        df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d').dt.date
        df.sort_values(by="Date", inplace=True)
    df_player_stats = pd.concat([df_player_stats21, df_player_stats22])
    df_gk_stats = pd.concat([df_gk_stats21, df_gk_stats22])
    if player_path.is_file() and gk_path.is_file():
        final_player_df = pd.read_csv("csv_files/2021_22_playerstats_epl.csv")
        final_player_df["Date"] = pd.to_datetime(final_player_df["Date"], format='%Y-%m-%d').dt.date
        final_player_df.sort_values(by="Date", inplace=True)
        if df_player_stats22["Date"].iloc[-1] > final_player_df["Date"].iloc[-1]:
            df_player_stats.to_csv("csv_files/2021_22_playerstats_epl.csv", index=False, header=True)
            df_gk_stats.to_csv("csv_files/2021_22_goalkeeperstats_epl.csv", index=False, header=True)
            print("Updated stats database")
    else:
        df_player_stats.to_csv("csv_files/2021_22_playerstats_epl.csv", index=False, header=True)
        df_gk_stats.to_csv("csv_files/2021_22_goalkeeperstats_epl.csv", index=False, header=True)
        print("Created stats database")

'''filters dataframe to find starting lineup for a given game'''
def starting_lineups(date, h_team, a_team, dataframe, ctrl=False):
    try:
        dataframe["Date"] = pd.to_datetime(dataframe["Date"], format="%Y-%m-%d").dt.date
        date = pd.to_datetime(date, format="%Y-%m-%d").date()
        home_data = dataframe[
            (dataframe["Date"] == date) & (dataframe["Venue"] == "Home") & (dataframe["Squad"] == h_team)]
        if not ctrl:
            home_data = home_data[home_data['Start'].isin(['Y', 'Y*'])]
        away_data = dataframe[
            (dataframe["Date"] == date) & (dataframe["Venue"] == "Away") & (dataframe["Squad"] == a_team)]
        if not ctrl:
            away_data = away_data[away_data['Start'].isin(['Y', 'Y*'])]
        if home_data.empty or away_data.empty:
            raise ValueError("No lineups found between teams on this date")
    except ValueError:
        logging.basicConfig(filename="logs/err_log.log", filemode="a", level=logging.INFO)
        logging.error("Unable to create game data for " + h_team + " " + a_team + " " + str(date) + " due to mising linuep info")
    return home_data, away_data

'''creates entry for a fixture, either called to create schema for a prediction or to help create the 
training schema'''
def create_feature_list(lineups):
    def_pos = ['CB', 'LB', 'WB', 'RB']
    mid_pos = ['CM', 'DM', 'LM', 'RM', 'AM']
    att_pos = ['FW', 'LW', 'RW']

    d_features = ['TklW', 'DribContest', 'SuccPress%', 'Int', 'Clr', 'Dis', 'TacklesDef3rd',
                  'Gls', 'Ast', 'KP', 'xG',
                  'xA', 'SoT']
    m_features = ['Cmp', 'PassAtt', 'Cmp%', 'Ast', 'xA', 'KP', 'DribShot', 'GCA', 'TklW',
                  'SuccPress%', 'Gls', 'xG',
                  'TacklesMid3rd', 'SoT']
    a_features = ['xG', 'SoT', 'Gls', 'CPA', 'Succ%', 'TacklesAtt3rd', 'GCA', 'Ast', 'xA', 'KP',
                  'Cmp', 'Int', 'Clr']
    g_features = ['GA', 'Saves', 'CS', 'PSxG', 'Cmp%']
    hsl = lineups[0]
    asl = lineups[1]
    home_keeper = lineups[2]
    away_keeper = lineups[3]
    hsl["Pos"] = hsl["Pos"].str[:2]
    home_defenders = hsl[hsl["Pos"].isin(def_pos)][d_features].agg(["mean"]).add_prefix("d_").reset_index()
    home_midfielders = hsl[hsl["Pos"].isin(mid_pos)][m_features].agg(["mean"]).add_prefix("m_").reset_index()
    home_attackers = hsl[hsl["Pos"].isin(att_pos)][a_features].agg(["mean"]).add_prefix("a_").reset_index()
    home_keeper = home_keeper[g_features].agg(["mean"]).add_prefix("g_").reset_index()

    asl["Pos"] = asl["Pos"].str[:2]
    away_defenders = asl[asl["Pos"].isin(def_pos)][d_features].agg(["mean"]).add_prefix("away_d_").reset_index()
    away_midfielders = asl[asl["Pos"].isin(mid_pos)][m_features].agg(["mean"]).add_prefix("away_m_").reset_index()
    away_attackers = asl[asl["Pos"].isin(att_pos)][a_features].agg(["mean"]).add_prefix("away_a_").reset_index()
    away_keeper = away_keeper[g_features].agg(["mean"]).add_prefix("away_g_").reset_index()
    schema_entry = pd.concat(
        [home_defenders, home_midfielders, home_attackers, home_keeper, away_defenders, away_midfielders,
         away_attackers, away_keeper], axis=1)
    schema_entry.drop(["index"], axis=1, inplace=True)
    if schema_entry.isnull().values.any():
        #doesn't raise exception as missing data gets filled in later in prediction_algorithms, just in the log.
        logging.basicConfig(filename="logs/err_log.log", filemode="a", level=logging.INFO)
        logging.info("Couldn't create schema entry for "+str(hsl["Squad"].iloc[0])+" on "+str(hsl["Date"].iloc[0]))
    schema_entry = [v for _, v in schema_entry.to_dict(orient="index").items()]
    return schema_entry

'''called to create the training schema, checks fixtures against JSON stream and iteratively
calls starting_lineups(), then writes out to final_schema.csv'''
def create_training_schema(date, start_date):
    df_player_stats = pd.read_csv("csv_files/2021_22_playerstats_epl.csv")
    df_gk_stats = pd.read_csv("csv_files/2021_22_goalkeeperstats_epl.csv")
    training_schema = []
    url_2020 = 'https://fixturedownload.com/feed/json/epl-2020'
    data_2020 = unpack_json_data(url_2020)
    data_2021 = unpack_json_data(url_2021)
    json_data = data_2020 + data_2021
    for n in json_data:
        json_data_date = pd.to_datetime(n["DateUtc"], format='%Y-%m-%d').date()
        if date >= json_data_date > start_date:
            hsl, asl = starting_lineups(json_data_date, n["HomeTeam"], n["AwayTeam"], df_player_stats)
            home_keeper, away_keeper = starting_lineups(json_data_date, n["HomeTeam"], n["AwayTeam"], df_gk_stats)
            lineups = [hsl, asl, home_keeper, away_keeper]
            [aggr_schema] = create_feature_list(lineups)
            aggr_schema["Home_goals_scored"] = n["HomeTeamScore"]
            aggr_schema["Away_goals_scored"] = n["AwayTeamScore"]
            aggr_schema["Home_team"] = n["HomeTeam"]
            aggr_schema["Away_team"] = n["AwayTeam"]
            aggr_schema["Date"] = n["DateUtc"]
            training_schema.append(aggr_schema)
    final_training_schema = pd.DataFrame(training_schema)
    final_training_schema.dropna(axis="rows", inplace=True)

    if schema_file.is_file():
        final_training_schema.to_csv(schema_file, mode='a', header=False, index=False)
    else:
        final_training_schema.to_csv(schema_file, index=False)

    return final_training_schema


if __name__ == '__main__':
    from datetime import date as dtd
    create_training_schema(dtd(2022, 4, 28), dtd(2020, 9, 11))
