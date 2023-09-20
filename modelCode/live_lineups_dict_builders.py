from pathlib import Path

import requests
import pandas as pd
import json

headers = {
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    "X-RapidAPI-Key": "6dd9177744msh3f463899aa5046ap14ec92jsn3b23538f5acc"
}

# used to create fixture database for the 2021/22 EPL season
def get_fixtures():
    url_fixtures = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring21 = {"league": "39", "season": "2021", "from": "2021-08-12", "to": "2022-05-22"}
    querystring20 = {"league": "39", "season": "2020", "from": "2020-09-12", "to": "2021-05-24"}

    response_fixtures21 = requests.request("GET", url_fixtures, headers=headers, params=querystring21).json()
    response_fixtures20 = requests.request("GET", url_fixtures, headers=headers, params=querystring20).json()

    list_of_dicts = []
    response_list = [response_fixtures20, response_fixtures21]
    for response in response_list:
        for item in response["response"]:
            fix_dict = {"fixt_id": item["fixture"]["id"], "home_team_id": item["teams"]["home"]["id"],
                        "away_team_id": item["teams"]["away"]["id"],
                        "date": pd.to_datetime(item["fixture"]["date"], format="%Y-%m-%d").date().strftime("%Y-%m-%d")}
            list_of_dicts.append(fix_dict)
    with open('json_files/fixtures.json', 'w') as fout:
        json.dump(list_of_dicts, fout)
    fout.close()

# creates team_id dictionary
def teamid_dbuilder():
    tid_dict21 = {}
    tid_dict20 = {}
    url_teams = "https://api-football-v1.p.rapidapi.com/v3/teams"
    querystring_teams21 = {"league": "39", "season": "2021"}
    querystring_teams20 = {"league": "39", "season": "2020"}
    response_teams21 = requests.request("GET", url_teams, headers=headers, params=querystring_teams21).json()
    response_teams20 = requests.request("GET", url_teams, headers=headers, params=querystring_teams20).json()
    for n in response_teams21["response"]:
        tid_dict21[n["team"]["name"]] = n["team"]["id"]
    for n in response_teams20["response"]:
        tid_dict20[n["team"]["name"]] = n["team"]["id"]
    keys21 = tid_dict21.keys()
    keys20 = tid_dict20.keys()
    difference = keys20 - keys21
    for n in difference:
        tid_dict21[n] = tid_dict20[n]

#weights averages for data given by most recently played month
def weight_averages(h_team, a_team, lineup_list, players_df, gk_df):
    hl_df = []
    al_df = []
    h_gk = []
    a_gk = []
    for temp_lineup in lineup_list:
        for player in temp_lineup:
            if player in set(players_df["Name"]):
                #these two if statements filter data to only keep data from current club,
                #prevents issues when player moves teams
                if player in lineup_list[0]:
                    filtered_df = players_df[(players_df["Squad"]) == h_team]
                elif player in lineup_list[1]:
                    filtered_df = players_df[(players_df["Squad"]) == a_team]
                else:
                    raise Exception("Malformed home/away values in df")
                last_game = filtered_df[(filtered_df["Name"]) == player].iloc[[-1]]
                player_series = filtered_df[(filtered_df["Name"]) == player]
            elif player in set(gk_df["Name"]):
                if player in lineup_list[2]:
                    filtered_df = gk_df[(gk_df["Squad"]) == h_team]
                elif player in lineup_list[3]:
                    filtered_df = gk_df[(gk_df["Squad"]) == a_team]
                else:
                    raise Exception("Malformed home/away values in player df")
                last_game = filtered_df[(filtered_df["Name"]) == player].iloc[[-1]]
                player_series = filtered_df[(filtered_df["Name"]) == player]
            else:
                raise ValueError("issue matching lineups to dataframes")
            player_series = player_series.set_index(["Date"])
            player_series.index = pd.to_datetime(player_series.index, format="%Y-%m-%d")
            player_series = player_series.resample('M').mean()
            player_series.dropna(inplace=True)
            # how data is weighted by most recent months
            most_recent_month = player_series.iloc[-1].mul(0.4)
            months_2_3 = pd.DataFrame(player_series.iloc[-3:-1].mean().mul(0.3))
            months_4_5 = player_series.iloc[-5:-3].mean().mul(0.15)
            months_6_8 = player_series.iloc[-8:-5].mean().mul(0.1)
            months_before_8 = player_series.iloc[:-8].mean().mul(0.05)
            entry = pd.concat([most_recent_month, months_2_3, months_4_5, months_6_8, months_before_8], axis=1).sum(
                axis=1)
            entry = pd.DataFrame(entry).T
            last_game.reset_index(drop=True, inplace=True)
            entry.reset_index(drop=True, inplace=True)
            entry_final = pd.concat([last_game[last_game.columns[0:9]], entry], axis=1)
            [entry_dict] = entry_final.to_dict("records")
            if entry_dict["Squad"] == h_team:
                if entry_dict["Pos"] == "GK":
                    h_gk.append(entry_dict)
                else:
                    hl_df.append(entry_dict)
            else:
                if entry_dict["Pos"] == "GK":
                    a_gk.append(entry_dict)
                else:
                    al_df.append(entry_dict)

    hl_df = pd.DataFrame(hl_df)
    al_df = pd.DataFrame(al_df)
    h_gk = pd.DataFrame(h_gk)
    a_gk = pd.DataFrame(a_gk)
    return hl_df, al_df, h_gk, a_gk

# when called returns weighted lineup data
def live_lineups(h_team, a_team, date):
    players_df = pd.read_csv("csv_files/2021_22_playerstats_epl.csv", encoding="UTF-8")
    gk_df = pd.read_csv("csv_files/2022_goalkeeperstats_epl.csv", encoding="UTF-8")
    df_2022 = pd.read_csv("csv_files/2022_playerstats_epl.csv", encoding="UTF-8")
    fixtures_path = Path("json_files/fixtures.json")
    team_ids = {'Manchester Utd': 33, 'Newcastle Utd': 34, 'Watford': 38, 'Wolves': 39, 'Liverpool': 40,
                'Southampton': 41, 'Arsenal': 42, 'Burnley': 44, 'Everton': 45, 'Leicester City': 46, 'Tottenham': 47,
                'West Ham': 48, 'Chelsea': 49, 'Manchester City': 50, 'Brighton': 51, 'Crystal Palace': 52,
                'Brentford': 55, 'Leeds Utd': 63, 'Aston Villa': 66, 'Norwich City': 71, 'Fulham': 36, 'West Brom': 60,
                'Sheffield Utd': 62}
    if not fixtures_path.is_file():
        get_fixtures()

    fixtures_json_file = open('json_files/fixtures.json')
    fixtures_json = json.load(fixtures_json_file)
    fixtures_json_file.close()
    date = pd.to_datetime(date, format="%Y-%m-%d").date().strftime("%Y-%m-%d")
    fixt_dict = next(
        (item for item in fixtures_json if item['home_team_id'] == team_ids[h_team] and item["date"] == date), None)
    if fixt_dict is None:
        raise ValueError("No Fixture between these teams on this date")
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups"
    querystring = {"fixture": str(fixt_dict["fixt_id"])}
    response = requests.request("GET", url, headers=headers, params=querystring)
    response = response.json()
    if response["results"] == 0:
        raise Exception("Lineups not released yet")
    starting_lineup = []
    keepers = []
    for n in response["response"]:
        for k in n["startXI"]:
            if k["player"]["pos"] == "G":
                keepers.append(k["player"]["name"])
            else:
                starting_lineup.append(k["player"]["name"])

    names_dict = {"Hee-Chan Hwang": "Hwang Hee-chan", "Jonny Otto": "Jonny Castro", "Samir": "Samir Santos",
                  "Ederson Moraes": "Ederson", "Emerson Royal": "Emerson", "Heung-min Son": "Son Heung-min",
                  "Gabriel Magalhães": "Gabriel Dos Santos", "Gabriel Martinelli": "Martinelli", "Nicolas N'Koulou": "Nicolas Nkoulou",
                  "Wilfredo Daniel Caballero": "Willy Caballero", "J. Pickford": "Jordan Pickford", "Cucurella": "Marc Cucurella",
                  "David de Gea Quintana": "David de Gea", "Romelu Lukaku Menama": "Romelu Lukaku", "Douglas Luiz Soares de Paulo": "Douglas Luiz"}
    for index, p in enumerate(starting_lineup):
        if p in names_dict.keys():
            starting_lineup[index] = names_dict[p]
    for index, p in enumerate(keepers):
        if p in names_dict.keys():
            keepers[index] = names_dict[p]

    splitname_pl = []
    splitname_pl_gks = []
    splitname_lineups = []
    splitname_gks = []
    [splitname_pl.append(name.split()) for name in players_df["Name"]]
    [splitname_lineups.append(name.split()) for name in starting_lineup]
    [splitname_pl_gks.append(name.split()) for name in gk_df["Name"]]
    [splitname_gks.append(name.split()) for name in keepers]
    final_lineups = []
    for starter in splitname_lineups:
        for index, name in enumerate(splitname_pl):
            if starter[-1] == name[-1] and starter[0][0] == name[0][0]:
                if players_df["Name"][index] not in final_lineups:
                    final_lineups.append(players_df["Name"][index])

    for starting_keeper in splitname_gks:
        for index, gk_name in enumerate(splitname_pl_gks):
            if starting_keeper[-1] == gk_name[-1] and starting_keeper[0][0] == gk_name[0][0]:
                if gk_df["Name"][index] not in final_lineups:
                    final_lineups.append(gk_df["Name"][index])

    final_lineups.remove("David Luiz")
    if len(final_lineups) != 22:
        raise Exception("issue parsing names given by API")
    lineup_list = [final_lineups[:10], final_lineups[10:20], [final_lineups[20]], [final_lineups[21]]]
    hl_df, al_df, h_gk, a_gk = weight_averages(h_team, a_team, lineup_list, players_df, gk_df)
    return hl_df, al_df, h_gk, a_gk

if __name__ == '__main__':
    get_fixtures()