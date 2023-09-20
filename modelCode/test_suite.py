import json
import unittest
import modelCode
import pandas as pd

'''testing suite using e2e an unit testing'''
class TestDataPreProcessing(unittest.TestCase):
    def test_live_lineups_success(self):
        df = pd.read_csv("csv_files/2021_22_playerstats_epl.csv")
        lineups = modelCode.starting_lineups("2020-09-12", "Fulham", "Arsenal", df)
        hl, al = lineups
        hl = list(hl["Name"])
        self.assertEqual(hl, ['Ivan Cavaleiro', 'Aboubakar Kamara', 'Tim Ream', 'Josh Onomah', 'Michael Hector',
                              'Joe Bryan', 'Denis Odoi', 'Harrison Reed', 'Tom Cairney', 'Neeskens Kebano'],
                         "Lineups not valid")

    def test_live_lineups_failure(self):
        df = pd.read_csv("csv_files/2021_22_playerstats_epl.csv")
        modelCode.starting_lineups("2022-01-01", "Fulham", "Arsenal", df)
        log = open("logs/err_log.log")
        for line in reversed(list(log)):
            most_recent_line = line.rstrip()
            break
        log.close()
        self.assertEqual(most_recent_line, "ERROR:root:Unable to create game data for Fulham Arsenal 2022-01-01 due to mising linuep info")

    def test_create_feature_list_success(self):
        df = pd.read_csv("csv_files/2021_22_playerstats_epl.csv")
        gk_df = pd.read_csv("csv_files/2021_22_goalkeeperstats_epl.csv")
        pl = modelCode.starting_lineups("2020-09-12", "Fulham", "Arsenal", df)
        gkl = modelCode.starting_lineups("2020-09-12", "Fulham", "Arsenal", gk_df)
        hl, al = pl
        hgk, agk = gkl
        lu = [hl, al, hgk, agk]
        fl = modelCode.create_feature_list(lu)
        correct_schema_file = open('json_files/correct_test_schema.json')
        correct_schema = json.load(correct_schema_file)
        correct_schema_file.close()
        self.assertEqual(correct_schema, fl)
    def test_create_feature_list_failure(self):
        log = open("logs/err_log.log")
        for line in reversed(list(log)):
            most_recent_line = line.rstrip()
            break
        print(most_recent_line)
        log.close()
        self.assertEqual(most_recent_line, "ERROR:root:Unable to create game data for Fulham Arsenal 2022-01-01 due to mising linuep info")


if __name__ == '__main__':
    unittest.main()
