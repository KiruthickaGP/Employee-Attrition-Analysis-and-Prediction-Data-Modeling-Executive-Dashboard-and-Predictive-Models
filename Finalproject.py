#importing the necessary libraries
#PART 1

import sklearn
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import traceback
import os 
import csv
import mysql.connector as sql
from mysql.connector import Error
from sqlalchemy import create_engine
from dash import Dash, html, dcc, Output, Input
import warnings
warnings.filterwarnings('ignore')
from urllib.parse import quote
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.metrics import accuracy_score

#PART 2
# Open the image file for the page icon
icon = Image.open("E:\\Guvidatascience\\Projects\\Final_project\\Finalproject.png")

# Set page configurations with background color
st.set_page_config(
    page_title="Employment attrition Analysis , Visualization and Prediction  | By Kiruthicka",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': """# This app is created by *Kiruthicka!*"""})

# Add background color using CSS
background_color = """
<style>
    body {
        background-color: #F7EBED;  /* Set background color to #F7EBED*/            #AntiqueWhite color
    }
    .stApp {
        background-color: #F7EBED; /* Set background-color for the entire app */
    }
</style>
"""
#AntiqueWhite color #F7EBED
st.markdown(background_color, unsafe_allow_html=True)


# CREATING OPTION MENU
with st.sidebar:
    selected = option_menu(None,["Home", "Extract and Transform", "Dashboard","Predictive analysis","Conclusion"],
        icons=["house-fill","tools","book","activity","layers"],
        default_index=2,
        orientation="Vertical",
        styles={
            "nav-link": {
                "font-size": "30px",
                "font-family": "Fira Sans",
                "font-weight": "Bold",
                "text-align": "left",
                "margin": "10px",
                "--hover-color": "#C1ADAE"#Grayish red
            },
            "icon": {"font-size": "30px"},
            "container": {"max-width": "6000px"},
            "nav-link-selected": {
                "background-color": "#c3909b",#Grey Pink
                "color": "Grey Pink",
            }
        }
    )




# HOME PAGE
# Open the image file for the YouTube logo
logo = Image.open("E:\\Guvidatascience\\Projects\\Final_project\\Finalproject.png")

# Define a custom CSS style to change text color
custom_style = """
<style>
    .black-text {
        color: black; /* Change text color to black */
    }
</style>
"""

   
# Apply the custom style
st.markdown(custom_style, unsafe_allow_html=True)

if selected == "Home":
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open("E://Guvidatascience//Projects//Final_project//Finalproject.png")
        st.image(image, width=500, caption='Image with Border', output_format='PNG', use_column_width=False)
        st.markdown("<style>div.stImage img {border-radius: 10px; border: 2px solid #008000;}</style>", unsafe_allow_html=True)

    with col2:
        st.markdown("## :green[**Technologies Used :**]")
        st.markdown("##### Python, easy OCR, Streamlit, SQL, Pandas.")

        st.markdown("## :green[**Overview :**]")
        st.markdown("##### Data Collection and Preparation")
        st.markdown("Gather historical employee data, clean it by handling missing values and outliers, and engineer features indicative of attrition risk.")

        st.markdown("##### Exploratory Data Analysis (EDA)")
        st.markdown("Explore data through statistical analysis, identifying key factors contributing to attrition.")

        st.markdown("##### Executive Dashboard")
        st.markdown("Design an interactive dashboard using tools like Tableau or Power BI, presenting key metrics, trends, and visualizations related to attrition.")

        st.markdown("##### Prediction Models")
        st.markdown("Select and train machine learning algorithms like logistic regression and decision trees, evaluating their performance and choosing the best model.")

        st.markdown("##### Recommendations")
        st.markdown("Identify significant variables impacting attrition, provide actionable recommendations to management, and prioritize interventions based on key factors.")

        st.markdown("##### Documentation and Presentation")
        st.markdown("Document the entire process, from data preparation to model building, and create a presentation summarizing findings, insights, and recommendations.")

#Extract and Transform
if selected == "Extract and Transform":
    st.title("Upload a file")

    def load_data(file):
        data = pd.read_csv(file)
        return data

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is None:
        st.info("Upload a file through config")
        st.stop()

    df = load_data(uploaded_file)
    st.dataframe(df)
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "Kimoni1710@?",
        "database": "attrition_db",
    }

    encoded_password = quote(db_config["password"])

    # Create a connection string
    connection_url = f'mysql+mysqlconnector://{db_config["user"]}:{encoded_password}@{db_config["host"]}/{db_config["database"]}'

    # Create SQLAlchemy engine
    engine = create_engine(connection_url)

    # Store the DataFrame in the SQL database
    df.to_sql(name="Attrition", con=engine, if_exists='replace', index=False)
    print("OK")

    st.success(f'Data uploaded to MySQL table: {"Attrition"}')


# Dashboard    

if selected == "Dashboard":

    st.title(":bar_chart: Employee Attrition analysis")
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

    fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
    
    if fl is not None:
        filename = fl.name
        st.write(filename)
        df = pd.read_csv(fl, encoding="ISO-8859-1")  # Use 'fl' directly as the file object
    else:
        st.warning("Please upload a file to view the dashboard.")
        st.stop()  # Stop execution if no file is uploaded

    

    st.header("Choose your filter: ")
    # Create for Department
    Department = st.multiselect("Pick your Department", df["Department"].unique())
    if not Department:
        df2 = df.copy()
    else:
        df2 = df[df["Department"].isin(Department)]

    # Create for JobRole
    JobRole = st.multiselect("Pick the JobRole", df2["JobRole"].unique())
    if not JobRole:
        df3 = df2.copy()
    else:
        df3 = df2[df2["JobRole"].isin(JobRole)]

    # Create for EducationField
    EducationField = st.multiselect("Pick the EducationField",df3["EducationField"].unique())

    # Filter the data based on Department, JobRole and EducationField

    if not Department and not JobRole and not EducationField:
        filtered_df = df
    elif not JobRole and not EducationField:
        filtered_df = df[df["Department"].isin(Department)]
    elif not Department and not EducationField:
        filtered_df = df[df["JobRole"].isin(JobRole)]
    elif JobRole and EducationField:
        filtered_df = df3[df["JobRole"].isin(JobRole) & df3["EducationField"].isin(EducationField)]
    elif Department and EducationField:
        filtered_df = df3[df["Department"].isin(Department) & df3["EducationField"].isin(EducationField)]
    elif Department and JobRole:
        filtered_df = df3[df["Department"].isin(Department) & df3["JobRole"].isin(JobRole)]
    elif EducationField:
        filtered_df = df3[df3["EducationField"].isin(EducationField)]
    else:
        filtered_df = df3[df3["Department"].isin(Department) & df3["JobRole"].isin(JobRole) & df3["EducationField"].isin(EducationField)]

    
    Department_df = filtered_df.groupby(by = ["Department"], as_index = False)["JobInvolvement"].sum()
    
    col1, col2 = st.columns((2))
    with col1:
        st.subheader("Department wise JobInvolvement")
        fig = px.bar(Department_df, x = "Department", y = "JobInvolvement", text = ['${:,.2f}'.format(x) for x in Department_df["JobInvolvement"]],
                    template = "seaborn")
        st.plotly_chart(fig,use_container_width=True, height = 200)

    with col2:
        st.subheader("Gender wise JobInvolvement")
        fig = px.pie(filtered_df, values = "JobInvolvement", names = "Gender", hole = 0.5)
        fig.update_traces(text = filtered_df["Gender"], textposition = "outside")
        st.plotly_chart(fig,use_container_width=True)

    cl1, cl2 = st.columns((2))

    with cl1:
        with st.expander("Department_ViewData"):
            st.write(Department_df.style.background_gradient(cmap="Blues"))
            csv = Department_df.to_csv(index = False).encode('utf-8')
            st.download_button("Download Data", data = csv, file_name = "Department.csv", mime = "text/csv",
                                help = 'Click here to download the data as a CSV file')

    with cl2:
        with st.expander("MaritalStatus_ViewData"):
            MaritalStatus= filtered_df.groupby(by = "MaritalStatus", as_index = False)["JobInvolvement"].sum()
            st.write(MaritalStatus.style.background_gradient(cmap="Oranges"))
            csv = MaritalStatus.to_csv(index = False).encode('utf-8')
            st.download_button("Download Data", data = csv, file_name = "MaritalStatus.csv", mime = "text/csv",
                            help = 'Click here to download the data as a CSV file')
            

   
    # Create a treemap
    # Replace 'your_file.csv' with the actual path to your CSV file
    csv_file_path = "E://Guvidatascience//Projects//Final_project//Attrition_Analysis.csv"

    # Load data from CSV file into a DataFrame
    filtered_df = pd.read_csv(csv_file_path)
    st.subheader("Hierarchical view of Sales using TreeMap")
    fig3 = px.treemap(filtered_df, path=["Department", "JobRole", "EducationField"], values="JobLevel", hover_data=["JobLevel"],
                        color="JobLevel")
    fig3.update_layout(width=800, height=650)
    st.plotly_chart(fig3, use_container_width=True)

    # Pie chart 1
    chart1, chart2 = st.columns(2)
    with chart1:
        st.subheader('YearsAtCompany wise MonthlyIncome')
        fig_pie1 = px.pie(filtered_df, values="MonthlyIncome", names="YearsAtCompany", template="plotly_dark")
        fig_pie1.update_traces(text=filtered_df["YearsAtCompany"], textposition="inside")
        st.plotly_chart(fig_pie1, use_container_width=True)

    # Pie chart 2
    with chart2:
        st.subheader('JobRole wise MonthlyIncome')
        fig_pie2 = px.pie(filtered_df, values="MonthlyIncome", names="JobRole", template="gridon")
        fig_pie2.update_traces(text=filtered_df["JobRole"], textposition="inside")
        st.plotly_chart(fig_pie2, use_container_width=True)

    # Scatter Plot
    data1 = px.scatter(filtered_df, x="TotalWorkingYears", y="PercentSalaryHike", size="PerformanceRating")
    data1['layout'].update(title="Relationship between TotalWorkingYears and PercentSalaryHike using Scatter Plot.",
                            titlefont=dict(size=20),
                            xaxis=dict(title="TotalWorkingYears", titlefont=dict(size=19)),
                            yaxis=dict(title="PercentSalaryHike", titlefont=dict(size=19)))
    st.plotly_chart(data1, use_container_width=True)

    # View Data Expander
    with st.expander("View Data"):
        st.write(filtered_df.iloc[:500, 1:20:2].style.background_gradient(cmap="Oranges"))

    # Download Original DataSet
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Data', data=csv, file_name="Data.csv", mime="text/csv")

#Predictive Analysis

# Model 1 :Neural network

if selected == "Predictive analysis":
    
    
    # Load data
    data = pd.read_excel("E://Guvidatascience//Projects//Final_project//Employee_attrition.xlsx")

    # Display the original data
    st.title("Employee Attrition Data")
    st.subheader("Original Data")
    st.dataframe(data)

    # Preprocess the data
    le = LabelEncoder()
    data['Attrition'] = le.fit_transform(data['Attrition'])
    st.title("Employee Attrition Data")
    
    st.title ("Model 1 : Neural network")
    # Display original data
    
    st.subheader("Attrition Data before transformation")
    st.dataframe(data)  # Use st.table(data) if you prefer a table layout
    

    # Label encoding
    le = LabelEncoder()
    data['Attrition'] = le.fit_transform(data['Attrition'])
    st.subheader("Attrition Data after transformation")
    st.dataframe(data)

    # Feature scaling
    inplist = data.columns[:-1]
    st.subheader("Attrition Data before scaling")
    st.dataframe(data)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy if needed
    data[inplist] = imputer.fit_transform(data[inplist])

    # Feature scaling (after imputation)
    scale = StandardScaler()
    data[inplist] = scale.fit_transform(data[inplist])
    st.subheader("Attrition Data after scaling")
    st.dataframe(data)

    # Feature scaling
    inplist = data.columns[:-1]
    imputer = SimpleImputer(strategy='mean')
    data[inplist] = imputer.fit_transform(data[inplist])
    scale = StandardScaler()
    data[inplist] = scale.fit_transform(data[inplist])

    # Split the data
    x = data.values[:, :-1]
    y = data.values[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=11)

    # Model Selection Dropdown
    selected_model = st.selectbox("Select Model", ["Neural Network", "Linear Regression", "k-Nearest Neighbors"])

    if selected_model == "Neural Network":
        # Neural Network Model
        st.title("Model 1: Neural Network")
        clf_nn = MLPClassifier(hidden_layer_sizes=3, activation='logistic', max_iter=150, solver='adam', learning_rate='constant', learning_rate_init=0.19)
        clf_nn.fit(x_train, y_train)
        ypred_nn = clf_nn.predict(x_test)

        # Display results in Streamlit
        st.subheader("Neural Network Classifier Analysis")
        st.write("Confusion Matrix:\n", confusion_matrix(y_test, ypred_nn))
        st.text("Classification Report:\n" + classification_report(y_test, ypred_nn))
        st.write("Accuracy:", accuracy_score(y_test, ypred_nn))
        st.line_chart(clf_nn.loss_curve_)

    elif selected_model == "Linear Regression":
        # Linear Regression Model
        st.title("Model 2: Linear Regression")
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)
        ypred_lr = regressor.predict(x_test)

        # Display results in Streamlit
        st.subheader("Linear Regression Analysis")
        st.write("Mean Squared Error:", mean_squared_error(y_test, ypred_lr))
        st.write("R-squared:", r2_score(y_test, ypred_lr))
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.regplot(x=y_test, y=ypred_lr, ax=ax)
        ax.set_xlabel("Actual Attrition")
        ax.set_ylabel("Predicted Attrition")
        ax.set_title("Linear Regression: Actual vs Predicted Attrition")
        st.pyplot(fig)

    else:
        # k-Nearest Neighbors Model
        st.title("Model 3: k-Nearest Neighbors")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train)
        ypred_knn = knn.predict(x_test)

        # Display results in Streamlit
        st.subheader("k-Nearest Neighbors Classifier Analysis")
        st.write("Confusion Matrix:\n", confusion_matrix(y_test, ypred_knn))
        st.text("Classification Report:\n" + classification_report(y_test, ypred_knn))
        st.write("Accuracy:", accuracy_score(y_test, ypred_knn))
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, ypred_knn), annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(plt.gcf())

    

    
    # Model Selection Dropdown
    selected_models = st.multiselect("Select Models to Compare", ["Neural Network", "Linear Regression", "k-Nearest Neighbors"])

    # Checkbox for comparison
    compare_checkbox = st.checkbox("Compare")

    # Check if the "Compare" checkbox is checked
    if compare_checkbox:

        results = pd.DataFrame(columns=['Model 1', 'Model 2', 'Metric', 'Value'])

        for i in range(len(selected_models)):
            for j in range(i + 1, len(selected_models)):
                model_1 = selected_models[i]
                model_2 = selected_models[j]

                if model_1 == "Neural Network" and model_2 == "k-Nearest Neighbors":
                    # Compare accuracy for Neural Network and k-Nearest Neighbors
                    clf_nn = MLPClassifier(hidden_layer_sizes=3, activation='logistic', max_iter=150, solver='adam', learning_rate='constant', learning_rate_init=0.19)
                    clf_nn.fit(x_train, y_train)
                    ypred_nn = clf_nn.predict(x_test)

                    knn = KNeighborsClassifier(n_neighbors=5)
                    knn.fit(x_train, y_train)
                    ypred_knn = knn.predict(x_test)

                    acc_nn = accuracy_score(y_test, ypred_nn)
                    acc_knn = accuracy_score(y_test, ypred_knn)

                    results = pd.concat([results, pd.DataFrame({'Model 1': ['Neural Network'], 'Model 2': ['k-Nearest Neighbors'], 'Metric': ['Accuracy'], 'Value': [acc_nn]})], ignore_index=True)
                    results = pd.concat([results, pd.DataFrame({'Model 1': ['k-Nearest Neighbors'], 'Model 2': ['Neural Network'], 'Metric': ['Accuracy'], 'Value': [acc_knn]})], ignore_index=True)

                elif model_1 == "Neural Network" and model_2 == "Linear Regression":
                    # Compare accuracy for Neural Network and Linear Regression
                    clf_nn = MLPClassifier(hidden_layer_sizes=3, activation='logistic', max_iter=150, solver='adam', learning_rate='constant', learning_rate_init=0.19)
                    clf_nn.fit(x_train, y_train)
                    ypred_nn = clf_nn.predict(x_test)

                    regressor = LinearRegression()
                    regressor.fit(x_train, y_train)
                    ypred_lr = regressor.predict(x_test)

                    acc_nn = accuracy_score(y_test, ypred_nn)
                    r2_lr = r2_score(y_test, ypred_lr)

                    results = pd.concat([results, pd.DataFrame({'Model 1': ['Neural Network'], 'Model 2': ['Linear Regression'], 'Metric': ['Accuracy'], 'Value': [acc_nn]})], ignore_index=True)
                    results = pd.concat([results, pd.DataFrame({'Model 1': ['Linear Regression'], 'Model 2': ['Neural Network'], 'Metric': ['R-squared'], 'Value': [r2_lr]})], ignore_index=True)

                elif model_1 == "Linear Regression" and model_2 == "k-Nearest Neighbors":
                    # Compare accuracy for Linear Regression and k-Nearest Neighbors
                    regressor = LinearRegression()
                    regressor.fit(x_train, y_train)
                    ypred_lr = regressor.predict(x_test)

                    knn = KNeighborsClassifier(n_neighbors=5)
                    knn.fit(x_train, y_train)
                    ypred_knn = knn.predict(x_test)

                    r2_lr = r2_score(y_test, ypred_lr)
                    acc_knn = accuracy_score(y_test, ypred_knn)

                    results = pd.concat([results, pd.DataFrame({'Model 1': ['Linear Regression'], 'Model 2': ['k-Nearest Neighbors'], 'Metric': ['R-squared'], 'Value': [r2_lr]})], ignore_index=True)
                    results = pd.concat([results, pd.DataFrame({'Model 1': ['k-Nearest Neighbors'], 'Model 2': ['Linear Regression'], 'Metric': ['Accuracy'], 'Value': [acc_knn]})], ignore_index=True)

        st.subheader("Comparison Results")
        st.table(results)


#Conclusion 

# Assuming 'selected' is a variable determined based on user input or some condition


if selected == "Conclusion":
        # Subtitle styles
        subtitle_style = 'font-size: 20px; font-weight: bold;'


        # Point 1
        st.subheader("📊 Key Factors Identified")
        st.text("")
        st.markdown("##### Our analysis identified key factors influencing employee attrition.")


        # Point 2
        st.subheader("🔍 Advanced Prediction Models")
        st.text("")
        st.markdown("##### Utilizing advanced prediction models, we pinpointed critical areas for intervention, emphasizing job satisfaction, workload, and training.")


        # Point 3
        st.subheader("🚀 Targeted Strategies for Enhancement")
        st.text("")
        st.markdown("##### Implementing targeted strategies in these areas can significantly enhance employee retention, mitigating project delays and reputation loss.")


        # Point 4
        st.subheader("📈 Informed Decision-Making Tools")
        st.text("")
        st.markdown("##### The project equips management with informed decision-making tools, fostering a more stable and productive work environment for the organization and its workforce.")