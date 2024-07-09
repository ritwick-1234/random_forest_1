
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('diabetes.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


st.sidebar.header('Hyperparameter Tuning')

n_estimators = st.sidebar.slider('Number of estimators', 1, 200, 100)
criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy', 'log_loss'))

max_depth_option = st.sidebar.selectbox('Max depth', options=['None', 'Specify'])
if max_depth_option == 'Specify':
    max_depth = st.sidebar.slider('Specify max depth', 1, 32, 16)
else:
    max_depth = None

min_samples_split = st.sidebar.slider('Min samples split', 2, 100, 2)
min_samples_leaf = st.sidebar.slider('Min samples leaf', 1, 200, 1)

max_features_type = st.sidebar.selectbox('Max features type', ('auto', 'sqrt', 'log2', 'all'))
if max_features_type == 'all':
    max_features = None
else:
    max_features = max_features_type

max_leaf_nodes = st.sidebar.slider('Max leaf nodes', 2, 200, None)
min_impurity_decrease = st.sidebar.slider('Min impurity decrease', 0.0, 0.5, 0.0)
bootstrap = st.sidebar.checkbox('Bootstrap', True)
oob_score = st.sidebar.checkbox('OOB Score', False)



if st.sidebar.button('Submit'):
   
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        oob_score=oob_score,
        
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

  
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {accuracy:.2f}')

    if oob_score:
        st.write(f'OOB Score: {model.oob_score_:.2f}')

   
