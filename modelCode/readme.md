Football Score Prediction Algorithm README:
prediction_algorithms.py is the main script, from there you can either run 
evaluation() or gs_model(). gs_model() is used for creating predictions with live
lineups from RAPID-API. evaluation() can be run with a number of different flags
depending on what you want to do, i.e. evaluate the control models, evaluate the 
hypothesis models, output predictions to betting DB etc. Web scraper in the repo 
doesn't work but recently found a way round FBref.com anti-scraping so will update
the function.
