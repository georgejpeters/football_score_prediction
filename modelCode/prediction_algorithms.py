from __future__ import print_function

import json
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import modelCode
from pathlib import Path
import timeit
from datetime import date as dtd

'''REALLY IMPORTANT NOTE: Odds scraper and parts of web-scraper use chromedriver, 
due to file size constraints with eBART hand in chrome driver had to be deleted from 
the project. Chromedriver needs to be in path with all the other scripts for some
functionality to work.'''

start = timeit.default_timer()

pd.options.mode.chained_assignment = None
pd.set_option("display.max_rows", None, "display.max_columns", None)

'''Used to create a home goals scored and away goals scored prediction using 5 different regression models
can also be used to evaluate a model and tune its hyperparameters'''


def prediction_algorithm(date, h_team, a_team, features, training_schema, model, ev=False, ctrl=False,
                         lineups_passed=False, tuning=False):
    # will need to change where feature list is sliced if final schema changes
    df_player_stats = pd.read_csv("csv_files/2021_22_playerstats_epl.csv")
    df_gk_stats = pd.read_csv("csv_files/2021_22_goalkeeperstats_epl.csv")
    home_features = features[:40] + features[85:90] + features[45:52]
    away_features = features[45:85] + features[40:45] + features[:7]
    if not ev:
        training_schema = training_schema[training_schema["Date"] < date]
    if not training_schema.empty:
        if h_team in set(training_schema.Home_team) or h_team in set(training_schema.Away_team):
            # defines training X and y
            home_X = training_schema[home_features].values
            home_y = training_schema["Home_goals_scored"]
            away_X = training_schema[away_features].values
            away_y = training_schema["Away_goals_scored"]
            # used to create lineup data
            if lineups_passed:
                hsl, asl, home_keeper, away_keeper = modelCode.live_lineups(h_team, a_team, date)
            else:
                if ctrl:
                    hsl, asl = modelCode.starting_lineups(date, h_team, a_team, df_player_stats, ctrl=True)
                    home_keeper, away_keeper = modelCode.starting_lineups(date, h_team, a_team, df_gk_stats, ctrl=True)
                else:
                    hsl, asl = modelCode.starting_lineups(date, h_team, a_team, df_player_stats)
                    home_keeper, away_keeper = modelCode.starting_lineups(date, h_team, a_team, df_gk_stats)
                lu_list = [list(hsl["Name"]), list(asl["Name"]), list(home_keeper["Name"]), list(away_keeper["Name"])]
                hsl, asl, home_keeper, away_keeper = modelCode.weight_averages(h_team, a_team, lu_list, df_player_stats,
                                                                               df_gk_stats)
            lineups = [hsl, asl, home_keeper, away_keeper]
            home_and_away_dict = modelCode.create_feature_list(lineups)
            home_and_away_schema = pd.DataFrame(home_and_away_dict)
            home_prev_data = home_and_away_schema[home_features]
            away_prev_data = home_and_away_schema[away_features]
            # if and elif check if either df has na values if so replaces with previous game stats,
            # prevents edge cases where teams lineups dont contain a certain position i.e. a striker
            if home_prev_data.isnull().values.any():
                fs_temp = pd.read_csv("csv_files/final_schema.csv")
                fs_temp["Date"] = pd.to_datetime(fs_temp["Date"], format='%Y-%m-%d').dt.date
                fs_temp = fs_temp[(fs_temp["Date"] <= date) & (fs_temp["Home_team"] == h_team)]
                fs_temp = fs_temp.iloc[[-1]]
                na_keys = home_prev_data.columns[home_prev_data.isna().any()].tolist()
                for key in na_keys:
                    home_prev_data[key] = fs_temp[key].values
            elif away_prev_data.isnull().values.any():
                fs_temp = pd.read_csv("csv_files/final_schema.csv")
                fs_temp["Date"] = pd.to_datetime(fs_temp["Date"], format='%Y-%m-%d').dt.date
                fs_temp = fs_temp[(fs_temp["Date"] <= date) & (fs_temp["Away_team"] == a_team)]
                fs_temp = fs_temp.iloc[[-1]]
                na_keys = away_prev_data.columns[away_prev_data.isna().any()].tolist()
                for key in na_keys:
                    away_prev_data[key] = fs_temp[key].values
            home_prev_data = home_prev_data.iloc[0]
            away_prev_data = away_prev_data.iloc[0]
            if not (training_schema[home_features].keys() == home_prev_data.keys()).all():
                raise Exception("schemas do not line up")
            home_prev_data = [np.array(home_prev_data, dtype=float)]
            away_prev_data = [np.array(away_prev_data, dtype=float)]

            if model == "LR":
                home_regr = linear_model.LinearRegression()
                away_regr = linear_model.LinearRegression()
            #regression models with tuned hyperparameters
            elif model == "RFR":
                home_regr = RandomForestRegressor(n_estimators=15, random_state=0, max_depth=5)
                away_regr = RandomForestRegressor(n_estimators=15, random_state=0, max_depth=5)
                # n_estimators=15, random_state=0, max_depth=5
            elif model == "DTR":
                home_regr = DecisionTreeRegressor(random_state=0, max_depth=6,
                                                  min_samples_split=10,
                                                  max_leaf_nodes=5, min_samples_leaf=20)
                away_regr = DecisionTreeRegressor(random_state=0, max_depth=6,
                                                  min_samples_split=10,
                                                  max_leaf_nodes=5, min_samples_leaf=20)
            elif model == "KNN":
                home_regr = KNeighborsRegressor(n_neighbors=4)
                away_regr = KNeighborsRegressor(n_neighbors=4)
            elif model == "SVR":
                home_regr = SVR(kernel="linear", epsilon=0.2)
                away_regr = SVR(kernel="linear", epsilon=0.2)
            else:
                raise ValueError("Invalid Model Input")
            #normalises data
            norm = MinMaxScaler().fit(home_X)
            home_X = norm.transform(home_X)
            home_prev_data = norm.transform(home_prev_data)
            away_X = norm.transform(away_X)
            away_prev_data = norm.transform(away_prev_data)
            home_regr.fit(home_X, home_y)
            away_regr.fit(away_X, away_y)
            #makes prediction for home and away goals scored
            ctrl_predicted_home_gs = home_regr.predict(home_prev_data)
            ctrl_predicted_away_gs = away_regr.predict(away_prev_data)
            if tuning:
                return ctrl_predicted_home_gs, ctrl_predicted_away_gs, home_X
            else:
                return ctrl_predicted_home_gs, ctrl_predicted_away_gs
        else:
            raise ValueError("Teams input not in DB")
    else:
        raise Exception("Training Schema Empty")


def gs_model(date, h_team, a_team, model):
    #checks if dataframes are formatted
    modelCode.format_dataframes()
    schema_path = Path("csv_files/final_schema.csv")
    datetime_date = pd.to_datetime(date, format="%d/%m/%Y").date()
    min_date = dtd.min
    player_df = pd.read_csv("csv_files/2021_22_playerstats_epl.csv")
    player_df["Date"] = pd.to_datetime(player_df["Date"], format="%Y-%m-%d").dt.date
    player_df.sort_values(by="Date", inplace=True)
    pdf_last_date = player_df["Date"].iloc[-1]
    if schema_path.is_file():
        training_schema = pd.read_csv("csv_files/final_schema.csv")
        training_schema["Date"] = pd.to_datetime(training_schema["Date"], format="%Y-%m-%d").dt.date
        training_schema.sort_values(by="Date", inplace=True)
        ts_last_date = training_schema["Date"].iloc[-1]
        if not training_schema.empty:
            # unfortunately not functional due to web scraper being broken
            # so never calls create training schema here
            if pdf_last_date > ts_last_date:
                print("Updating Final Schema")
                modelCode.create_training_schema(pdf_last_date, ts_last_date)

        else:
            print("Final Schema empty, populating...")
            modelCode.create_training_schema(pdf_last_date, min_date)
    else:
        print("Final Schema Being Created")
        modelCode.create_training_schema(pdf_last_date, min_date)
    training_schema = pd.read_csv("csv_files/final_schema.csv")
    schema_keys = training_schema.keys()
    not_used = ['Home_team', 'Home_goals_scored', "Away_goals_scored", "Away_team", "Date"]
    features = [item for item in schema_keys if item not in not_used]
    training_schema["Date"] = pd.to_datetime(training_schema["Date"], format="%Y-%m-%d").dt.date
    # calls prediction algorithm to make prediction
    ctrl_predicted_home_gs, ctrl_predicted_away_gs = prediction_algorithm(datetime_date, h_team,
                                                                          a_team, features,
                                                                          training_schema, model,
                                                                          lineups_passed=True)
    print(str(h_team) + ' ' + str(np.round(ctrl_predicted_home_gs))[1:-1] + ' - ' + str(
        a_team) + ' ' + str(np.round(ctrl_predicted_away_gs))[1:-1] + " " + model)

'''evaluation functionality, can be called to tune hyperparameter, create predictions
for betting analysis or call control model'''
def evaluation(model, tuning=False, betting=False, ctrl=False):
    modelCode.format_dataframes()
    betting_data = []
    h_y_pred = []
    a_y_pred = []
    split_date = dtd(2022, 2, 1)
    final_schema = pd.read_csv("csv_files/final_schema.csv")
    schema_keys = final_schema.keys()
    # removes features not used in training schema
    not_used = ['Home_team', 'Home_goals_scored', "Away_goals_scored", "Away_team", "Date"]
    features = [item for item in schema_keys if item not in not_used]
    final_schema["Date"] = pd.to_datetime(final_schema["Date"], format="%Y-%m-%d").dt.date
    final_schema.sort_values(by="Date", inplace=True)
    query_schema = final_schema[(final_schema["Date"] >= split_date)]
    training_schema = final_schema[(final_schema["Date"] < split_date)]
    # iterates through given schema to create predictions for betting analysis
    if betting:
        for entry in query_schema.itertuples():
            temp_dict = {}
            [predicted_home_gs], [predicted_away_gs] = prediction_algorithm(entry.Date, entry.Home_team,
                                                                            entry.Away_team, features,
                                                                            training_schema, model,
                                                                            ev=split_date)

            predicted_home_gs = 0 if predicted_home_gs < 0 else predicted_home_gs
            predicted_away_gs = 0 if predicted_away_gs < 0 else predicted_away_gs
            predicted_home_gs = int(np.round(predicted_home_gs))
            predicted_away_gs = int(np.round(predicted_away_gs))
            predicted_and_actual_score = [(predicted_home_gs, predicted_away_gs),
                                          (int(entry.Home_goals_scored), int(entry.Away_goals_scored))]
            passed_date = entry.Date.strftime("%Y-%m-%d")
            temp_dict["teams"] = [entry.Home_team, entry.Away_team]
            temp_dict["date"] = [passed_date]
            temp_dict["predicted_score"] = [[str(predicted_home_gs) + ":" + str(predicted_away_gs)], [
                str(int(entry.Home_goals_scored)) + ":" + str(int(entry.Away_goals_scored))]]
            betting_data.append(temp_dict)
        return betting_data

    else:
        for entry in query_schema.itertuples():
            # calls prediction algorithm with tuning flag to find tuning hyperparameters
            if tuning:
                [predicted_home_gs], [predicted_away_gs], home_X = prediction_algorithm(entry.Date, entry.Home_team,
                                                                                        entry.Away_team, features,
                                                                                        training_schema, model,
                                                                                        ev=split_date, tuning=True)
            # calls control predicted goals scored model
            elif ctrl:
                [predicted_home_gs], [predicted_away_gs] = prediction_algorithm(entry.Date, entry.Home_team,
                                                                                entry.Away_team, features,
                                                                                training_schema, model,
                                                                                ev=split_date, ctrl=True)
            # calls hypothesis goals scored model
            else:
                [predicted_home_gs], [predicted_away_gs] = prediction_algorithm(entry.Date, entry.Home_team,
                                                                                entry.Away_team, features,
                                                                                training_schema, model,
                                                                                ev=split_date)

            predicted_home_gs = 0 if predicted_home_gs < 0 else predicted_home_gs
            predicted_away_gs = 0 if predicted_away_gs < 0 else predicted_away_gs
            h_y_pred.append(int(np.round(predicted_home_gs)))
            a_y_pred.append(int(np.round(predicted_away_gs)))
    # uses gridsearch cv to find optimal hyperparamters using predictions made earlier in function
    if tuning:
        if model == "DTR":
            dtr = DecisionTreeRegressor(random_state=100)
            param_grid = {"criterion": ["mse", "absolute_error"],
                          "min_samples_split": [10, 20, 40],
                          "max_depth": [2, 6, 8],
                          "min_samples_leaf": [20, 40, 100],
                          "max_leaf_nodes": [5, 20, 100],
                          }
            grid_cv_dtm = GridSearchCV(dtr, param_grid, cv=5)
            grid_cv_dtm.fit(home_X, list(training_schema["Home_goals_scored"]))
            # print("R-Squared::{}".format(grid_cv_dtm.best_score_))
            print("Best Hyperparameters::\n{}".format(grid_cv_dtm.best_params_))
        elif model == "RFR":
            param_grid = {'bootstrap': [True, False], 'max_depth': [5, 10, None], 'max_features': ['auto', 'log2'],
                          'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]}
            rfr = RandomForestRegressor(random_state=1)

            grid = GridSearchCV(estimator=rfr, param_grid=param_grid,
                                cv=3, n_jobs=1, verbose=0, return_train_score=True)
            grid.fit(home_X, list(training_schema["Home_goals_scored"]))
            print(grid.best_params_)
        elif model == "KNN":
            knn = KNeighborsRegressor()
            param_grid = [{'n_neighbors': [2, 3, 4, 5, 6], 'weights': ['uniform', 'distance']}]
            grid = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3, scoring='r2')
            grid.fit(home_X, list(training_schema["Home_goals_scored"]))
            print(grid.best_estimator_)
            print(grid.best_params_)
        elif model == "SVR":
            param_grid = [{'kernel': ['rbf']},
                          {'kernel': ['linear']}]
            svr = SVR()
            grid = GridSearchCV(svr, param_grid=param_grid, verbose=0, n_jobs=-4, cv=4,
                                scoring='r2')
            grid.fit(home_X, list(training_schema["Home_goals_scored"]))
            print('Best Kernel:', grid.best_estimator_.kernel)

    print("Evaluation performed on a training set of", len(training_schema), " fixtures and a testing set of",
          len(query_schema), "fixtures")
    h_mae = metrics.mean_absolute_error(list(query_schema["Home_goals_scored"]), h_y_pred)
    a_mae = metrics.mean_absolute_error(list(query_schema["Away_goals_scored"]), a_y_pred)
    h_rmse = metrics.mean_squared_error(list(query_schema["Home_goals_scored"]), h_y_pred, squared=False)
    a_rmse = metrics.mean_squared_error(list(query_schema["Away_goals_scored"]), a_y_pred, squared=False)
    h_r2 = metrics.r2_score(list(query_schema["Home_goals_scored"]), h_y_pred)
    a_r2 = metrics.r2_score(list(query_schema["Away_goals_scored"]), a_y_pred)
    # print("Home: MAE: " + str(h_mae), "RMSE: " + str(h_rmse), "R2: " + str(h_r2) + " " + model)
    # print("Away: MAE: " + str(a_mae), "RMSE: " + str(a_rmse), "R2: " + str(a_r2) + " " + model)
    print("HOME:", model, "&", h_mae, "&", h_rmse, "&", h_r2)
    print("AWAY:", model, "&", a_mae, "&", a_rmse, "&", a_r2)

'''when called iterates through models to format betting data'''
def create_prediction_json_data():
    model_list = ["LR", "DTR", "SVR", "RFR", "KNN"]
    dict_of_json_data = {}
    for m in model_list:
        r_dict = evaluation(m, betting=True)
        print(r_dict)
        dict_of_json_data[m] = r_dict
    with open("json_files/betting_preds.json", "w") as f:
        json.dump(dict_of_json_data, f)
        f.close()


if __name__ == '__main__':
    model_list = ["LR", "DTR", "SVR", "RFR", "KNN"]
    for m in model_list:
        gs_model("10/05/2022", "Aston Villa", "Liverpool", m)
        # evaluation(m)
    # create_prediction_json_data()
    stop = timeit.default_timer()
    print('Time: ', stop - start)
