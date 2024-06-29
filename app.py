# import streamlit as st
# import pickle

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from  sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.linear_model import LogisticRegression



# st.markdown(
#     """
#     <h1 style='text-align: center;'>DIABETES PREDICTION Using Logistic Regression</h1>
#     """,
#     unsafe_allow_html=True
# )
# st.markdown(
#     """
#     <h4 style='text-align: center;'>Analysis of the dataset</h4>
#     """,
#     unsafe_allow_html=True
# )
# df=pd.read_csv("diabetes.csv")
# markdown_title = "### **CSV File Upload and Display**"
# st.markdown(markdown_title)


# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:

#     data_set = df


#     st.write("### Dataset")
#     html_string = f"<h4><b>The shape of the dataset is:</b> {df.shape}</h4>"
#     html_string_1=f" <h4>The size of the csv file is {uploaded_file.size} byte</h4>"
#     st.write( html_string_1,unsafe_allow_html=True)

#     st.write(html_string, unsafe_allow_html=True)

#     st.write(df)
# st.sidebar.markdown("<h3 style='text-align:center;'>See Options</h3>",unsafe_allow_html=True)
# option = st.sidebar.selectbox(
#     "**See bar chart for each column**",
#     ('Hide Bar Charts','Pregnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','DiabetesPedigreeFunction','Age' ,'show all together')
# )
# if option=='Hide Bar Charts':
#     pass
# elif option == 'Pregnancies':
#     st.write("Bar Chart for Pregnancies")
#     st.bar_chart(df['Pregnancies'])
# elif option == 'Glucose':
#     st.write("Bar Chart for Glucose")
#     st.bar_chart(df['Glucose'])
# elif option == 'Blood Pressure':
#     st.write("Bar Chart for Blood Pressure")
#     st.bar_chart(df['BloodPressure'])
# elif option == 'Skin Thickness':
#     st.write("Bar Chart for Skin Thickness")
#     st.bar_chart(df['SkinThickness'])
# elif option == 'Insulin':
#     st.write("Bar Chart for Insulin")
#     st.bar_chart(df['Insulin'])
# elif option == 'BMI':
#     st.write("Bar Chart for BMI")
#     st.bar_chart(df['BMI'])
# elif option == 'DiabetesPedigreeFunction':
#     st.write("Bar Chart for Diabetes Pedigree Function")
#     st.bar_chart(df['DiabetesPedigreeFunction'])
# elif option == 'Age':
#     st.write("Bar Chart for Age")
#     st.bar_chart(df['Age'])
# elif option == 'show all together':
    
#     st.bar_chart(df)
    

# data_frame = pd.DataFrame(df)


# # show_stats = st.sidebar.checkbox("Show Statistics")


# # if show_stats:
# #     st.write("Summary Statistics:")
# #     st.write(data_frame.describe())
# #     show_stats = st.sidebar.checkbox("Show Statistics")

# # show_stats_1 = st.sidebar.checkbox("See contribution of columns")
# # if show_stats_1:
# #     features = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
# #             'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
# #     contributions = [0.221898, 0.466581, 0.065068, 0.074752, 0.130548, 0.292695, 0.173844, 0.238356]

# # # Plotting
# #     plt.figure(figsize=(8, 8))
# #     plt.pie(contributions, labels=features, autopct='%1.1f%%', startangle=140)
# #     plt.title('Contribution of Features to Predict Outcome')
# #     plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# #     plt.show()
# import streamlit as st
# import matplotlib.pyplot as plt


# features = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
#             'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
# contributions = [0.221898, 0.466581, 0.065068, 0.074752, 0.130548, 0.292695, 0.173844, 0.238356]


# show_stats = st.sidebar.checkbox("Show Statistics")

# if show_stats:
#     st.write("Summary Statistics:")
#     st.write(data_frame.describe())



# show_stats_1 = st.sidebar.checkbox("See contribution of columns", key="see_contributions")

# if show_stats_1:
#     # Plotting
#     plt.figure(figsize=(8, 8))
#     plt.pie(contributions, labels=features, autopct='%1.1f%%', startangle=140)
#     plt.title('Contribution of Features to Predict Outcome')
#     plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#     st.pyplot(plt)  # Display the plot in Streamlit
#     st.title("Diabetic or Not")
# with open('diabetes_part_1.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)


# # Function to predict diabetes
# def predict_diabetes(data):
#     # Preprocess input data if needed (e.g., scaling)
#     # Make predictions
#     prediction = model.predict(data)
   
#     return prediction


# # Load the pre-trained model


# # Streamlit app title
# st.sidebar.title('Diabetes Prediction')

# # Define radio button options
# option = st.sidebar.radio("Select an option", ('Predict Diabetes',))

# # Main content area
# if option == 'Predict Diabetes':
#     st.title('Diabetes Prediction')

#     # Create a form for user input
#     with st.form("input_form"):
#         st.write("Fill in the details to predict diabetes:")
        
#         # Input fields for user to enter data
#         pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
#         glucose = st.number_input("Glucose", min_value=0)
#         blood_pressure = st.number_input("BloodPressure", min_value=0)
#         skin_thickness = st.number_input("SkinThickness", min_value=0)
#         insulin = st.number_input("Insulin", min_value=0)
#         bmi = st.number_input("BMI", min_value=0.0)
#         diabetes_pedigree = st.number_input("DiabetesPedigreeFunction", min_value=0.000,step=0.0001)
#         age = st.number_input("Age", min_value=0, step=1)

#         # Submit button to make prediction
#         submitted = st.form_submit_button("Predict")

#         if submitted:
#             # Create a DataFrame from user inputs
#             input_data = pd.DataFrame({
#                 'Pregnancies': [pregnancies],
#                 'Glucose': [glucose],
#                 'BloodPressure': [blood_pressure],
#                 'SkinThickness': [skin_thickness],
#                 'Insulin': [insulin],
#                 'BMI': [bmi],
#                 'DiabetesPedigreeFunction': [diabetes_pedigree],
#                 'Age': [age]
#             })

#             # Make prediction using the model
#             prediction = predict_diabetes(input_data)

#             # Display prediction result and print
#             if prediction[0] == 1:
#                 st.error("The person is predicted to be diabetic.")
#                 st.write("Prediction: Diabetic")
#             else:
#                 st.success("The person is predicted to be non-diabetic.")
#                 st.write("Prediction: Non-diabetic")
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('diabetes.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar for hyperparameter tuning
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
# max_samples = st.sidebar.slider('Max samples', 1, X_train.shape[0], None)

# Add a submit button
if st.sidebar.button('Submit'):
    # Initialize and fit the RandomForestClassifier
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

    # Display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {accuracy:.2f}')

    # Display OOB score if applicable
    if oob_score:
        st.write(f'OOB Score: {model.oob_score_:.2f}')

    # Display the decision trees
    # st.write(f'Number of trees: {len(model.estimators_)}')
    # st.write('Decision Trees:')
    # for i, tree in enumerate(model.estimators_):
    #     st.write(f'Tree {i + 1}:')
    #     st.write(tree)
