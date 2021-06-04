# import necessary libraries
from model_functions import *
# from models import create_classes
import os
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

#################################################
# Database Setup
#################################################

# from flask_sqlalchemy import SQLAlchemy
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '') or "sqlite:///db.sqlite"

# # Remove tracking modifications
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)

# NFL = create_classes(db)

# create route that renders index.html template


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # name = request.form["nflform"]
        HLW = [int(request.form["homelast"])]
        VLW = [int(request.form["awaylast"])]
        HTWL = [int(request.form["teamwin"])]
        HWS = [int(request.form["homewin"])]
        VWS = [int(request.form["awaywin"])]
        HT = [request.form["hometeam"]]
        AT = [request.form["awayteam"]]
        print(HLW)
        build_model() 
        results=predict_user_input(HT, AT, HLW, VLW, HWS, VWS, HTWL)
    else:
        results=None
    return render_template("form.html", results=results) 


if __name__ == "__main__":
    app.run()
