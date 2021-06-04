import pandas as pd

# Read and clean excel file into DF
def read_clean():
    years = list(range(2017,2018))
    dfs = []
    for year in years:
        df = pd.read_csv('NFL_{}.csv'.format(year))
        df = df.dropna(subset=['Week'])
        post_season = ['WildCard','Division','ConfChamp','SuperBowl']
        post_season_df = df[df['Week'].isin(post_season)]
        df = pd.concat([df,post_season_df]).drop_duplicates(keep=False)
        dfs.append(df)
    read_clean.NFL_2000_2016 = pd.DataFrame()
    for df in dfs:
        read_clean.NFL_2000_2016 = pd.concat([read_clean.NFL_2000_2016,df])
    read_clean.NFL_2000_2016 = read_clean.NFL_2000_2016.rename(columns={'Unnamed: 5':'Home/Away'}).reset_index(drop=True)
#     print(read_clean.NFL_2000_2016)

# Feature Engineering: Which team was the home team and which team was the away team?
def home_away():
    def home_team (row):
        if row['Home/Away'] != '@':
            return row['Winner/tie']
        else:
            return row['Loser/tie']
    def away_team (row):
        if row['Home/Away'] != '@':
            return row['Loser/tie']
        else:
            return row['Winner/tie']

    read_clean.NFL_2000_2016['Home Team'] = read_clean.NFL_2000_2016.apply(lambda row: home_team(row), axis=1)
    read_clean.NFL_2000_2016['Away Team'] = read_clean.NFL_2000_2016.apply(lambda row: away_team(row), axis=1)
    # NFL_2000_2016['Away Team']
#     print(read_clean.NFL_2000_2016)

# Feature Engineering: HomePts vs VisitorPts
def points():
    def home_pts (row):
        if row['Home/Away'] != '@':
            return row['Pts']
        else:
            return row['Pts.1']
    def away_pts (row):
        if row['Home/Away'] != '@':
            return row['Pts.1']
        else:
            return row['Pts']
    read_clean.NFL_2000_2016['HomePts'] = read_clean.NFL_2000_2016.apply(lambda row: home_pts(row), axis=1)
    read_clean.NFL_2000_2016['VisitorPts'] = read_clean.NFL_2000_2016.apply(lambda row: away_pts(row), axis=1)
#     print(read_clean.NFL_2000_2016)

# Create target class
def target_class():
    read_clean.NFL_2000_2016['HomeWin'] = read_clean.NFL_2000_2016['VisitorPts'] < read_clean.NFL_2000_2016['HomePts']
    # Class Values
    target_class.y_true = read_clean.NFL_2000_2016['HomeWin'].values
#     print(read_clean.NFL_2000_2016)

# Feature Engineering: Did teams win their previous game?
def prev_win():
    read_clean.NFL_2000_2016['HomeLastWin'] = False
    read_clean.NFL_2000_2016['VisitorLastWin'] = False

    from collections import defaultdict
    won_last = defaultdict(int)

    for index, row in read_clean.NFL_2000_2016.iterrows():
        home_team = row['Home Team']
        visitor_team = row['Away Team']
        row["HomeLastWin"] = won_last[home_team]
        row["VisitorLastWin"] = won_last[visitor_team]
        read_clean.NFL_2000_2016.iloc[index] = row
        # Set current win
        won_last[home_team] = row["HomeWin"]
        won_last[visitor_team] = not row['HomeWin']
#     print(read_clean.NFL_2000_2016)

# Feature Engineering: Win streaks.
def win_streaks():
    read_clean.NFL_2000_2016["HomeWinStreak"]=0
    read_clean.NFL_2000_2016["VisitorWinStreak"]=0
    from collections import defaultdict
    win_streak = defaultdict(int)

    for index, row in read_clean.NFL_2000_2016.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Away Team"]
        row["HomeWinStreak"] = win_streak[home_team]
        row["VisitorWinStreak"] = win_streak[visitor_team]
        read_clean.NFL_2000_2016.iloc[index] = row
        # Set current win
        if row["HomeWin"]:
            win_streak[home_team] +=1
            win_streak[visitor_team] = 0
        else:
            win_streak[home_team] = 0
            win_streak[visitor_team] += 1
    # print(read_clean.NFL_2000_2016)

# Feature Engineering: Did the home team win the last game between the two teams?
def home_team_win_last():
    import collections
    last_match_winner = collections.defaultdict(int)
    
    def home_team_won_last(row):
        home_team = row["Home Team"]
        visitor_team = row["Away Team"]

        # Sort for consistent ordering
        teams = tuple(sorted([home_team,visitor_team]))
        result = 1 if last_match_winner[teams] == row["Home Team"] else 0
        # Update record for next encounter
        winner = row["Home Team"] if row["HomeWin"] else row["Away Team"]

        last_match_winner[teams] = winner

        return result
    read_clean.NFL_2000_2016["HomeTeamWonLast"] = read_clean.NFL_2000_2016.apply(home_team_won_last, axis=1)
#     print(read_clean.NFL_2000_2016)

# OneHotEncoding
def OHE():
    OHE.NFL_model = read_clean.NFL_2000_2016[["Home Team","Away Team","HomeLastWin","VisitorLastWin","HomeWinStreak","VisitorWinStreak","HomeTeamWonLast"]]
    OHE.NFL_model = pd.get_dummies(OHE.NFL_model)
#     for i in OHE.NFL_model.columns:
#         print(i)

def create_X_all():
    import numpy as np
    create_X_all.X_all = np.hstack([OHE.NFL_model])

# def test_models():
#     import pandas as pd
#     from sklearn.linear_model import LogisticRegression
#     from sklearn import tree
#     from sklearn.naive_bayes import GaussianNB
#     from sklearn.neighbors import KNeighborsClassifier
#     from sklearn.ensemble import RandomForestClassifier
#     from xgboost import XGBClassifier
#     from sklearn.datasets import make_classification
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
#     import numpy as np

#     # features = NFL_2000_2016[["HomeLastWin","VisitorLastWin","HomeWinStreak","VisitorWinStreak","HomeTeamWonLast"]]
#     test_models.X_all = np.hstack([OHE.NFL_model])

#     class Classifiers():
#         def __init__(self, x_train, x_test, y_train):
#             self.x_train = x_train
#             self.y_train = y_train
#             self.x_test = x_test
#         def logistic_regression(self):
#             model = LogisticRegression()
#             model.fit(self.x_train, self.y_train)
#             predicted = model.predict(self.x_test)
#             return predicted
#         def decision_tree(self):
#             model = tree.DecisionTreeClassifier(criterion='gini')
#             model.fit(self.x_train, self.y_train)
#             predicted = model.predict(self.x_test)
#             return predicted
#         def naive_bayes(self):
#             model = GaussianNB()
#             model.fit(self.x_train, self.y_train)
#             predicted = model.predict(self.x_test)
#             return predicted
#         def knn(self):
#             model = KNeighborsClassifier(n_neighbors=6)
#             model.fit(self.x_train, self.y_train)
#             predicted= model.predict(self.x_test)
#             return predicted
#         def random_forest(self):
#             model= RandomForestClassifier()
#             model.fit(self.x_train, self.y_train)
#             predicted = model.predict(self.x_test)
#             return predicted
#         def xgboost(self):
#             model = XGBClassifier()
#             model.fit(self.x_train, self.y_train)
#             predicted = model.predict(self.x_test)
#             return predicted
    # def test():
    #     X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #     data_to_classify = Classifiers(X_train, X_test, y_train)
    #     d = {'original_result': y_test, 'predicted_result': data_to_classify.random_forest()}
    #     df = pd.DataFrame(data=d)
    #     df["correct_prediction"] = (df.original_result == df.predicted_result)
    #     print(df, f"Accuracy Score: \n{df.correct_prediction.value_counts(normalize=True)}")
    # test()

    # def compare_Classifiers_accuracy(X_train, X_test, y_train, y_test):
    #     outcome_accuracy = {
    #         "logistic_regression": accuracy_score(y_test, Classifiers(X_train, X_test, y_train).logistic_regression()),
    #         "decision_tree": accuracy_score(y_test, Classifiers(X_train, X_test, y_train).decision_tree()),
    #         "naive_bayes": accuracy_score(y_test, Classifiers(X_train, X_test, y_train).naive_bayes()),
    #         "knn": accuracy_score(y_test, Classifiers(X_train, X_test, y_train).knn()),
    #         "random_forest": accuracy_score(y_test, Classifiers(X_train, X_test, y_train).random_forest()),
    #         "xgboost": accuracy_score(y_test, Classifiers(X_train, X_test, y_train).xgboost()),
    #     }
    #     return outcome_accuracy
    # def main():
    #     seed = 1
    # #     X_all, y_true = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    #     X_train, X_test, y_train, y_test = train_test_split(create_X_all.X_all, target_class.y_true, test_size=0.33, random_state=seed)
    #     # Validation
    #     # X_validation_train, X_validation, y_validation_train, y_validation = train_test_split(X_train, y_train, test_size=0.33, random_state=seed) 
    #     print(compare_Classifiers_accuracy(X_train, X_test, y_train, y_test))
    #     # print(compare_Classifiers_accuracy(X_validation_train,X_validation))
    # main()

# Building Chosen Model: Random Forest
def build_RF():
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    seed = 1
    X_train, X_test, y_train, y_test = train_test_split(create_X_all.X_all, target_class.y_true, test_size=0.33, random_state=seed)

    model= RandomForestClassifier()
    model.fit(X_train, y_train)
    # predicted = model.predict(X_test)
    # print(X_all)

# Create user input from front end
# HT = ['New England Patriots']
# AT = ['Kansas City Chiefs']
# HLW = [0]
# VLW = [0]
# HWS = [0]
# VWS = [0]
# HTWL = [0]
def user_input(HT, AT, HLW, VLW, HWS, VWS, HTWL):
    user_input.NFL_model_ui = read_clean.NFL_2000_2016[["Home Team","Away Team","HomeLastWin","VisitorLastWin","HomeWinStreak","VisitorWinStreak","HomeTeamWonLast"]]
    user_input.user_input = pd.DataFrame({'Home Team':HT,'Away Team':AT,'HomeLastWin':HLW,'VisitorLastWin':VLW,'HomeWinStreak':HWS,'VisitorWinStreak':VWS,'HomeTeamWonLast':HTWL})
    user_input.NFL_model_ui = pd.concat([user_input.NFL_model_ui,user_input.user_input], axis = 0)
    user_input.NFL_model_ui = pd.get_dummies(user_input.NFL_model_ui)
    user_input.NFL_model_ui = pd.DataFrame(user_input.NFL_model_ui.iloc[-1]).T
    print(user_input.user_input)
# Test Prediction
def predict(AT, HT):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    seed = 1
    X_train, X_test, y_train, y_test = train_test_split(create_X_all.X_all, target_class.y_true, test_size=0.33, random_state=seed)

    model= RandomForestClassifier()
    model.fit(X_train, y_train)
    predicted = model.predict(user_input.NFL_model_ui)
    if predicted == [False]:
        print(f'{AT} will win.')
        return f'{AT} will win.'
    else:
        print(f'{HT} will win.')
        return f'{HT} will win.'

def build_model():
    read_clean()
    home_away()
    points()
    target_class()
    prev_win()
    win_streaks()
    home_team_win_last()
    OHE()
    create_X_all()
    build_RF()

def predict_user_input(HT, AT, HLW, VLW, HWS, VWS, HTWL):
    # try:
    user_input(HT, AT, HLW, VLW, HWS, VWS, HTWL)
    return predict(AT, HT)
    # except:
    #     print('There was a problem making a prediction. Please make sure you filled out the form correctly and try again.')

# build_model()
# predict_user_input(HT, AT, HLW, VLW, HWS, VWS, HTWL)