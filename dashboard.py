import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Create general path to reach our resources
path_resources="C:/Users/quent/Desktop/Formation_OCR/Projets/Projet_7/dashboard_resources/"


# Loading data (set with cache, mean it will not be reloaded if the input variables are not changed)
# This function will load all the tables used in the dashboard

@st.cache
def load_data(path_resources):


	path_df = path_resources +"data.csv"
	path_predict= path_resources +"cust_pred.csv"
	path_average = path_resources +"average.csv"
	path_features = path_resources + "shap_features.pickle"

	dataframe = pd.read_csv(path_df)
	dataframe.drop(columns='TARGET',inplace=True)

	predi = pd.read_csv(path_predict)
	average = pd.read_csv(path_average,index_col='Unnamed: 0')

	with open(path_features, 'rb') as file:
		features=pickle.load(file)


	customer = dataframe['SK_ID_CURR']
	customer=customer.astype('int64')
	cust_examples = str(list(customer[0:4].values)).replace('[','').replace(']','')

	return dataframe, customer, cust_examples, average, predi, features

# Prediction function is only used when a user changes the value of one feature
@st.cache
def prediction(path_resources, id_cust,data,modified=False, feature =None,value= None):
	path_std =path_resources+"scaler.pickle"
	path_model= path_resources+"model.pickle"
	path_th= path_resources+"threshold.pickle"

	with open(path_std, 'rb') as file:
		std=pickle.load(file)
	with open(path_model, 'rb') as file:
		model=pickle.load(file)
	with open(path_th, 'rb') as file:
		th=pickle.load(file)

	id_cust=int(id_cust)


	X = data[data['SK_ID_CURR'] == id_cust]

	if modified == True :
		X[feature]=value

	X = X.values
	X_scaled=std.transform(X)

	proba = model.predict_proba(X_scaled)

	if proba[0][1]>th:
		return 1, proba
	else :
		return 0, proba

#This function prepares the table which is required for to draw the plots
@st.cache
def get_graph_val(customer,average,dataframe,modified=False, feature =None,value= None):
	graph_val = average.copy()
	# st.write(average)

	graph_val.insert(0,'customer',dataframe.loc[dataframe['SK_ID_CURR']==int(customer)].values[0])

	if modified == True :
		graph_val.loc[feature,'customer']=value
	
	
	return graph_val


# Display plots (depending on the variables)
def graphes(customer, average,dataframe,features,nb_disp,modified=False,feature =None,value= None):
    '''Create a subplot with a number of features displayed given by the user'''

    ax_col = math.ceil(np.sqrt(nb_disp))
    ax_rows = math.ceil(nb_disp/ax_col)
    nb_graph =ax_rows*ax_col

    f, ax = plt.subplots(ax_rows, ax_col, figsize=(10,10), sharex=False)
    axs=ax.ravel()
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    
    df = get_graph_val(customer,average,dataframe,modified,feature,value)

   
    liste_cols = df.columns
    for i in range(nb_disp):

        sns.despine(ax=None, left=True, bottom=True, trim=False)
        sns.barplot(y = df.loc[features[i],:].values,
                   x = liste_cols,
                   ax = axs[i])
        sns.axes_style("white")

        plt.sca(axs[i])
        plt.title(list_feature[i])
        plt.xticks(rotation=45)

    if nb_disp < nb_graph:
    	for i in range(1,(nb_graph-nb_disp)+1):
    			axs[-i].axis('off')




    st.pyplot(f)
 
    return True


#------------------------------------------------------------------------------------------------------------------
#                                         Dashboard Creation Section
#------------------------------------------------------------------------------------------------------------------

# Get all tables from loading function 
dataframe,customers,customer_examples, average, predi, features = load_data(path_resources)

#------------------------------------------------------------------------------------------------------------------
#                                        Header Section (always displayed)
#------------------------------------------------------------------------------------------------------------------
st.title("Dashboard projet_7: Implémentez un modèle de scoring")
st.subheader("Quentin Stepniewski - OpenClassrooms DataScientist - Novembre 2020")

st.subheader("Prédictions de la capacité d'un client à rembourser son prêt")
id_input = st.text_input('Veuillez saisir l\'identifiant du client:', )

#------------------------------------------------------------------------------------------------------------------
#                                                 Body Section 
#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
#                                         Case when : No input is provided
#------------------------------------------------------------------------------------------------------------------
if id_input == '':
	st.write('Exemples de clients : '+customer_examples)

#------------------------------------------------------------------------------------------------------------------
#                                         Case when : Wrong input is provided
#------------------------------------------------------------------------------------------------------------------
elif (int(id_input) not in customers.values): 
	st.write('Exemples de clients : '+customer_examples)

	st.markdown('<style>p{color: red;}</style>', unsafe_allow_html=True)


	st.write(
		'**Client non reconnu**\n'
		'Veuillez réessayer\n')
	st.write(customers)

#------------------------------------------------------------------------------------------------------------------
#                                         Case when : Valid customer ID is entered
#------------------------------------------------------------------------------------------------------------------
else:

#  ======================================================================
#    SideBar Section (always displayed when valid customer is entered)
#  ======================================================================

	st.sidebar.header("Modifier le profil client")
	list_feature = features[:6]
	to_modify = ['Aucune']+list_feature
	feature_to_modify = st.sidebar.selectbox('Quelle caractéristique souhaitez-vous modifiier ?', to_modify)

#  ======================================================================
#    Case when feature to be modified = "Aucune"
#  ======================================================================

# Here we just display the already calculated values of prediction 
	if feature_to_modify == "Aucune":

		with st.spinner('Calcul du score du client...'):

			class_cust = int(predi[predi['SK_ID_CURR']==int(id_input)]['predict'].values[0])
			proba_state = predi[predi['SK_ID_CURR']==int(id_input)]['proba'].values[0]
			classe_reelle = int(predi[predi['SK_ID_CURR']==int(id_input)]['TARGET'].values[0])


			if class_cust == 1:
				etat = 'client à risque'
				st.markdown('<style>p{color: orange;}</style>', unsafe_allow_html=True)
			else:
				etat = 'client peu risqué'

			#Display Prediction

			classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')

			chaine = 'Prédiction : **' + etat +  '** avec **' + str(round(proba_state*100)) + '%** de risque de défaut (classe réelle : '+str(classe_reelle) + ')'

		st.markdown(chaine)

		st.subheader("Observation des caractéristiques impactant le plus la prédiction:")

		nb_display = st.selectbox('Combien de caractéristiques voulez-vous afficher ?', [2,3,4,5,6])

		graphes(id_input,average,dataframe,features,nb_display)




		st.subheader("Définition des groupes")
		st.markdown("\
		\n\
		* Customer : la valeur pour le client considéré\n\
		* mean_value : valeur moyenne pour l'ensemble des clients\n\
		* non_default_customer : valeur moyenne pour l'ensemble des clients sans défaut de paiement\n\
		* default_customer: valeur moyenne pour l'ensemble des clients présentant un défaut de paiement\n\
		")

#  ======================================================================
#    Case when feature to be modified is different from "Aucune"
#  ======================================================================
	else:


		# Setup the slider value in the side bar (min, max, default value and the step)
		mini = dataframe[feature_to_modify].min()
		maxi = dataframe[feature_to_modify].max()
		default_val = dataframe[dataframe['SK_ID_CURR']==int(id_input)][feature_to_modify].values[0]

		if (mini,maxi)==(0,1):
			step = 1
		else : 
			step = (maxi - mini)/20


		update_value = st.sidebar.slider(label = 'Nouvelle valeur (valeur d\'origine : {} )'.format(np.around(default_val,2)),
            min_value = float(mini),
            max_value = float(maxi),
            value = float(default_val),
			step = float(step))


        # ********************************************************************
        #  Case when value to be updated is still equal to the default value
        # ********************************************************************

        # Here is basically a copy/paste of the above section as no changes are required
		if update_value == default_val :
			with st.spinner('Calcul du score du client...'):
				class_cust = int(predi[predi['SK_ID_CURR']==int(id_input)]['predict'].values[0])
				proba_state = predi[predi['SK_ID_CURR']==int(id_input)]['proba'].values[0]
				classe_reelle = int(predi[predi['SK_ID_CURR']==int(id_input)]['TARGET'].values[0])


				if class_cust == 1:
					etat = 'client à risque'
					st.markdown('<style>p{color: orange;}</style>', unsafe_allow_html=True)
				else:
					etat = 'client peu risqué'

				#Display prediction 

				classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')

				chaine = 'Prédiction : **' + etat +  '** avec **' + str(round(proba_state*100)) + '%** de risque de défaut (classe réelle : '+str(classe_reelle) + ')'

			st.markdown(chaine)

			st.subheader("Observation des caractéristiques impactant le plus la prédiction:")

			nb_display = st.selectbox('Combien de caractéristiques voulez-vous afficher ?', [2,3,4,5,6])

			graphes(id_input,average,dataframe,features,nb_display)



			st.subheader("Définition des groupes")
			st.markdown("\
			\n\
			* Customer : la valeur pour le client considéré\n\
			* mean_value : valeur moyenne pour l'ensemble des clients\n\
			* non_default_customer : valeur moyenne pour l'ensemble des clients sans défaut de paiement\n\
			* default_customer: valeur moyenne pour l'ensemble des clients présentant un défaut de paiement\n\
			")



        # ********************************************************************
        #  Case when value to be updated different from the default value
        # ********************************************************************

        # Here we need to proceed with a new prediction and update one of the plots
		else : 
			with st.spinner('Calcul du score du client...'):
				class_cust, proba = prediction(path_resources,id_input,dataframe,True,feature_to_modify,update_value)

				if class_cust == 1:
					etat = 'client à risque'
					st.markdown('<style>p{color: orange;}</style>', unsafe_allow_html=True)
				else:
					etat = 'client peu risqué'

				proba_state = proba[0][1]

				#affichage de la prédiction

				classe_reelle = int(predi[predi['SK_ID_CURR']==int(id_input)]['TARGET'].values[0])
				
				classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')

				chaine = 'Prédiction : **' + etat +  '** avec **' + str(round(proba_state*100)) + '%** de risque de défaut (classe réelle : '+str(classe_reelle) + ')'

			st.markdown(chaine)

			st.subheader("Observation des caractéristiques impactant le plus la prédiction:")

			nb_display = st.selectbox('Combien de features voulez-vous afficher ', [2,3,4,5,6])

			graphes(id_input,average,dataframe,features,nb_display,True,feature_to_modify,update_value)




			st.subheader("Définition des groupes")
			st.markdown("\
			\n\
			* Customer : la valeur pour le client considéré\n\
			* mean_value : valeur moyenne pour l'ensemble des clients\n\
			* non_default_customer : valeur moyenne pour l'ensemble des clients sans défaut de paiement\n\
			* default_customer: valeur moyenne pour l'ensemble des clients présentant un défaut de paiement\n\
			")

