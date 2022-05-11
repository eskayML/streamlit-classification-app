import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
np.random.seed(42)

# plt.style.use('seaborn')
st.title("Perform different classification algorithms ")
st.write("""&copy;Dev Eskay""")
test_size = st.sidebar.slider("Select Test Size (%) ", 5,30)

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
# st.write(dataset_name)
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "Random Forest", "SVM",'GBM'))


def get_dataset(dataset_name):
	if dataset_name == "Iris":
		data = datasets.load_iris()
	elif dataset_name == "Breast Cancer":
		data = datasets.load_breast_cancer()
	else:
		data = datasets.load_wine()
	X,y = data.data , data.target
	return X,y



X,y = get_dataset(dataset_name)
st.write(f"""###### Dataset in use: {dataset_name}""" )
st.write("Shape of the dataset:", X.shape)
st.write("Number of Categories(classes):", len(np.unique(y)))




def add_parameter_ui(clf_name):
	params = {}
	if clf_name == "KNN":
		K = st.sidebar.slider("K",1,15)
		params["K"] = K
	elif clf_name == "SVM":
		C = st.sidebar.slider("C", 0.01,10.0)
		params["C"] = C
	elif clf_name == 'GBM':
		max_depth = st.sidebar.slider("max_depth", 2,16)
		n_estimators = st.sidebar.slider('n_estimators', 1,100)
		learning_rate = st.sidebar.slider('learning_rate',0.01,1.0)

		params["max_depth"] = max_depth
		params['n_estimators'] = n_estimators
		params['learning_rate'] = learning_rate
	else:
		max_depth = st.sidebar.slider("max_depth", 2,16)
		n_estimators = st.sidebar.slider('n_estimators', 1,100)

		params["max_depth"] = max_depth
		params['n_estimators'] = n_estimators
	return params


params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
	if clf_name == "KNN":
		clf = KNeighborsClassifier(n_neighbors = params['K'])
	elif clf_name == "SVM":
		clf = SVC(C = params['C'])
	elif clf_name == 'GBM':
		clf  = GradientBoostingClassifier(
			max_depth = params["max_depth"],
			n_estimators = params['n_estimators'],
			learning_rate  = params['learning_rate']
			)
	else:
		clf = RandomForestClassifier(
		max_depth = params["max_depth"],
		n_estimators = params['n_estimators'],
		
		)
	return clf



clf = get_classifier(classifier_name, params)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state  = 2022, test_size = round(test_size/100, 1)  )
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
st.markdown('### MODEL METRICS')
st.write("Model Accuracy: ", accuracy_score(y_test,y_pred))