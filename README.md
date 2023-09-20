Football Score Prediction Algorithm README:
prediction_algorithms.py is the main script, from there you can either run 
evaluation() or gs_model(). gs_model() is used for creating predictions with live
lineups from RAPID-API. evaluation() can be run with a number of different flags
depending on what you want to do, i.e. evaluate the control models, evaluate the 
hypothesis models, output predictions to betting DB etc. Currently, web scraper does
not work due to FBref.com blocking all web scrapers as of April 26 2022. 
IMPORTANT NOTE: Odds scraper and parts of web-scraper use chromedriver, 
due to file size constraints chrome driver had to be deleted from the project.
Chromedriver needs to be in path with all the other scripts to work.
