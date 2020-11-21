from flask import Flask, render_template, request, flash, redirect,jsonify
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from forms import PredictionForm

import pickle

app = Flask(__name__)


app.config['SECRET_KEY'] = "%-9qb-s_5%_6=%o!w@y7^#c3b^ef#*!v393wwhgkrv+n04fk6_"

#------------------------------------------------------------------------------------------------------------------
#                                         Load all required resources
#------------------------------------------------------------------------------------------------------------------

path_resources="C:/Users/quent/Desktop/Formation_OCR/Projets/Projet_7/dashboard_resources/"

path_std =path_resources+"scaler.pickle"
path_model= path_resources+"model.pickle"
path_th= path_resources+"threshold.pickle"
path_df = path_resources +"data.csv"

with open(path_std, 'rb') as file:
    std=pickle.load(file)
with open(path_model, 'rb') as file:
    model=pickle.load(file)
with open(path_th, 'rb') as file:
    th=pickle.load(file)

dataframe = pd.read_csv(path_df)

def prediction(id_cust,data):

    with open(path_std, 'rb') as file:
    	std=pickle.load(file)
    with open(path_model, 'rb') as file:
    	model=pickle.load(file)

    id_cust=int(id_cust)


    X = data[data['SK_ID_CURR'] == id_cust]
    X.drop(columns='TARGET',inplace=True)
    X_scaled=std.transform(X)

    

    proba = model.predict_proba(X_scaled)

    if proba[0][1]>th:
    	return 1, proba
    else :
    	return 0, proba


#------------------------------------------------------------------------------------------------------------------
#                                         Original Page function
#------------------------------------------------------------------------------------------------------------------
    
@app.route("/", methods=['GET', 'POST'])
def index():
	form = PredictionForm()

	if form.validate_on_submit():
		form_id = request.form['ID']

		return redirect('/dashboard/'+form_id)


	return render_template('predict.html', title='pred', form=form)


 
#------------------------------------------------------------------------------------------------------------------
#                                         Prediction  Page function
#------------------------------------------------------------------------------------------------------------------

@app.route('/dashboard/<id_client>',methods=['GET'])
def dashboard(id_client):
    id_client=id_client
    predi, proba = prediction(id_client, dataframe)
    dict_final = {
        'prediction' : int(predi),
        'proba_1' : float(proba[0][1])
        }
    print('Nouvelle Prédiction : \n', dict_final)

    # return 'Id : {}'.format(id_client)

    return jsonify(dict_final)




if __name__ == "__main__":
    app.run(debug=True)