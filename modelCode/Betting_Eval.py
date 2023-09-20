import pandas as pd
import json
from datetime import date as dtd
import matplotlib.pyplot as plt
import seaborn as sns
#pd.set_option("display.max_rows", None, "display.max_columns", None)
'''this script is run directly to find net earnings over a period with each model and graphs it'''
with open('json_files/betting_preds.json') as fixtures_json_file:
    predictions_data = json.load(fixtures_json_file)
    fixtures_json_file.close()
with open("json_files/final_odds_dict.json") as odds_file:
    odds_data = json.load(odds_file)
    odds_file.close()
odds_df = pd.DataFrame(odds_data)
odds_df["date"] = pd.to_datetime(odds_df["date"], format='%Y-%m-%d').dt.date
odds_df = odds_df[(odds_df["date"] >= dtd(2022, 2, 1)) & (odds_df["date"] <= dtd(2022, 4, 17))]
#print(odds_df.columns.tolist())
# odds_df = odds_df.iloc[::-1]
final_winnings_list = {}
for key, value in predictions_data.items():
    dataframe = pd.DataFrame(value)
    #print(odds_df["odds_data"])
    # print(dataframe["teams"])
    dataframe["teams"] = list(map(sorted, list(dataframe["teams"])))
    dataframe = dataframe.sort_values("teams")
    odds_df['teams'] = list(map(sorted, list(odds_df['teams'])))
    odds_df['length'] = odds_df['teams'].apply(lambda x: len(x))
    cond = odds_df['length'] == 2
    odds_df = odds_df[cond]
    odds_df = odds_df.sort_values("teams")
    total=0
    stake=1
    win_count=0
    winnings_list=[]
    for index, entry in dataframe.iterrows():
        [date] = entry["date"]
        date = pd.to_datetime(date).date()
        selected = odds_df.loc[odds_df["date"] == date]
        if (list(entry["teams"]) in list(selected["teams"])):
            teams = list(entry["teams"])
            odds_row = selected[selected['teams'].apply(lambda x: set(teams).issubset(x))]
            odds_data=odds_row['odds_data'].iloc[0]
            predicted_score = entry["predicted_score"][0]
            actual_score = entry["predicted_score"][1]
            if predicted_score == actual_score:
                [actual_score] = actual_score
                odds = odds_data[actual_score]
                odds = odds.split("/")
                winnings=(stake*int(odds[0]))/int(odds[1])
                total+=winnings
                winnings_list.append(winnings)
                win_count+=1
            else:
                total=total-stake
        winnings_list.append(total)
    final_winnings_list[key] = winnings_list
    print(total, key)
    print("win count", win_count)
sns.set_style("darkgrid")
p = sns.lineplot(data=final_winnings_list)
p.set_xlabel("Bets Made", fontsize = 15)
p.set_ylabel("Net Earnings", fontsize = 15)
