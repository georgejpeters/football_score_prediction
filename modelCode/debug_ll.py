import json
import time
import requests
import pandas as pd

'''this script was used to investigate the number of name discrepancies between the database
and the names provided by rapid-api'''
def get_player_data_from_api():
    list_of_json = []
    url = "https://api-football-v1.p.rapidapi.com/v3/players"
    headers = {
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
        "X-RapidAPI-Key": "6dd9177744msh3f463899aa5046ap14ec92jsn3b23538f5acc"
    }
    for i in range(1, 41):
        querystring = {"league": "39", "season": "2021", "page": str(i)}

        response = requests.request("GET", url, headers=headers, params=querystring)

        list_of_json.append(response.text)
        time.sleep(3)

    with open("json_files/json_players.json", "w") as f:
        json.dump(list_of_json, f)
        f.close()
def load_player_data():
    name_list = []
    with open("json_files/json_players.json") as f:
        player_data = json.load(f)
    for d in player_data:
        d = json.loads(d)
        for element in d["response"]:
            name_list.append(element["player"]["name"])
    name_list = set(name_list)
    df = pd.read_csv("csv_files/2022_playerstats_epl.csv")
    gk_df = pd.read_csv("csv_files/2022_goalkeeperstats_epl.csv")
    df_set = set(df["Name"])
    gk_df_set = set(df["Name"])
    names = name_list - df_set
    names = [" ".join(s.split()) for s in names]
    splitname_pl = []
    splitname_pl_gks = []
    splitname_lineups = []
    splitname_gks = []
    [splitname_pl.append(name.split()) for name in df_set]
    [splitname_lineups.append(name.split()) for name in names]
    [splitname_pl_gks.append(name.split()) for name in gk_df_set]
    for starter in splitname_lineups:
        for index, name in enumerate(splitname_pl):
            if starter[-1] == name[-1] and starter[0][0] == name[0][0]:
                names.remove(" ".join(starter))

    for starting_keeper in names:
        for index, gk_name in enumerate(splitname_pl_gks):
            if starting_keeper[-1] == gk_name[-1] and starting_keeper[0][0] == gk_name[0][0]:
                names.remove(" ".join(starting_keeper))
    splitname_pl = []
    splitname_pl_gks = []
    splitname_lineups = []
    splitname_gks = []
    [splitname_pl.append(name.split()) for name in df_set]
    [splitname_lineups.append(name.split()) for name in names]
    [splitname_pl_gks.append(name.split()) for name in gk_df_set]
    for starter in splitname_lineups:
        for index, name in enumerate(splitname_pl):
            if starter[-1] == name[-1]:
                print(" ".join(starter))
                #names.remove(" ".join(starter))
    print(names)
    print(len(names))
    dict = {}
if __name__ == '__main__':
    load_player_data()