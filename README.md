# NFL_Prediction

Data from the 2017 NFL season was scraped from https://www.pro-football-reference.com/ and data was feature engineered from that data. The data fed into the model was:

- Did the home team win their last game?
- Did the visiting team win their last game?
- What is the win streak of the home team?
- What is the win streak of the visiting team?
- Did the home team beat the visiting team last time they played?
- The names of all the teams.

These features were encoded using get_dummies and then fed into the model. The model we ultimately ended up going with was a random forest classifier that had an accuracy of 65%. A front-end website was built and deployed to Heroku that allows users to input those features above for the model to evaluate future games. A prediction is returned after hitting the submit button.

![Image description](https://github.com/sebastiandifrancesco/NFL_Prediction/blob/main/NFL_Prediction_Website.PNG)

The website is located at:

https://dashboard.heroku.com/apps/nfl-prediction2021
