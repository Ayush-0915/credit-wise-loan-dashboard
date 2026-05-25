import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Credit Wise Loan - Approval Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare data"""
    df = pd.read_csv("loan_approval_data.csv")
    
    # Handle missing values
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numericals_cols = df.select_dtypes(include=["float64"]).columns
    
    num_imp = SimpleImputer(strategy="mean")
    df[numericals_cols] = num_imp.fit_transform(df[numericals_cols])
    
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])
    
    # Drop Applicant ID
    df = df.drop("Applicant_ID", axis=1)
    
    # Encoding
    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])
    
    # One-hot encode
    cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", 
            "Property_Area", "Gender", "Employer_Category"]
    cols = [col for col in cols if col in df.columns]
    
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)
    
    df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)
    
    return df, ohe

@st.cache_data
def train_models(df):
    """Train all models"""
    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    results = {}
    
    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    y_pred_log = log_model.predict(X_test_scaled)
    models['Logistic Regression'] = (log_model, scaler)
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_log),
        'precision': precision_score(y_test, y_pred_log),
        'recall': recall_score(y_test, y_pred_log),
        'f1': f1_score(y_test, y_pred_log),
        'cm': confusion_matrix(y_test, y_pred_log)
    }
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    models['KNN'] = (knn, scaler)
    results['KNN'] = {
        'accuracy': accuracy_score(y_test, y_pred_knn),
        'precision': precision_score(y_test, y_pred_knn),
        'recall': recall_score(y_test, y_pred_knn),
        'f1': f1_score(y_test, y_pred_knn),
        'cm': confusion_matrix(y_test, y_pred_knn)
    }
    
    # Naive Bayes
    naive = GaussianNB()
    naive.fit(X_train_scaled, y_train)
    y_pred_naive = naive.predict(X_test_scaled)
    models['Naive Bayes'] = (naive, scaler)
    results['Naive Bayes'] = {
        'accuracy': accuracy_score(y_test, y_pred_naive),
        'precision': precision_score(y_test, y_pred_naive),
        'recall': recall_score(y_test, y_pred_naive),
        'f1': f1_score(y_test, y_pred_naive),
        'cm': confusion_matrix(y_test, y_pred_naive)
    }
    
    return models, results, X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Title and Sidebar
st.title("🏦 Credit Wise Loan Approval Dashboard")
st.markdown("---")

# Load data
df, ohe = load_and_prepare_data()
models, results, X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler = train_models(df)

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["🎯 Prediction", "📊 Model Performance", "📈 Data Analysis", "ℹ️ About"])

if page == "🎯 Prediction":
    st.header("Loan Approval Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Applicant Information")
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=25000)
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0)
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=35)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed"])
    
    with col2:
        st.subheader("Loan & Property Details")
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=100000)
        loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=360)
        dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
        savings = st.number_input("Savings ($)", min_value=0, value=10000)
        collateral_value = st.number_input("Collateral Value ($)", min_value=0, value=50000)
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Additional Info")
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=4, value=0)
        education_level = st.selectbox("Education Level", ["Not Graduate", "Graduate"])
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    with col4:
        st.subheader("Employment & Loan")
        employer_category = st.selectbox("Employer Category", ["Government", "Private"])
        loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Business", "Home"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=2)
    
    # Prepare prediction data
    if st.button("🔮 Predict Loan Approval", use_container_width=True, key="predict"):
        # Create input dataframe with same structure as training data
        input_data = pd.DataFrame({
            'Applicant_Income': [applicant_income],
            'Coapplicant_Income': [coapplicant_income],
            'Age': [age],
            'Credit_Score': [credit_score],
            'Existing_Loans': [existing_loans],
            'DTI_Ratio': [dti_ratio],
            'Savings': [savings],
            'Collateral_Value': [collateral_value],
            'Loan_Amount': [loan_amount],
            'Loan_Term': [loan_term],
            'Education_Level': [education_level],
            'Employment_Status': [employment_status],
            'Marital_Status': [marital_status],
            'Loan_Purpose': [loan_purpose],
            'Property_Area': [property_area],
            'Gender': [gender],
            'Employer_Category': [employer_category],
            'Dependents': [dependents]
        })
        
        # Encode education level
        le = LabelEncoder()
        le.fit(["Not Graduate", "Graduate"])
        input_data['Education_Level'] = le.transform(input_data['Education_Level'])
        
        # One-hot encode categorical variables
        categorical_cols = ['Employment_Status', 'Marital_Status', 'Loan_Purpose', 'Property_Area', 'Gender', 'Employer_Category']
        encoded_input = ohe.transform(input_data[categorical_cols])
        encoded_df = pd.DataFrame(encoded_input, columns=ohe.get_feature_names_out(categorical_cols))
        
        input_data = pd.concat([input_data.drop(columns=categorical_cols), encoded_df], axis=1)
        
        # Ensure all columns match training data
        missing_cols = set(X.columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        
        input_data = input_data[X.columns]
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make predictions with all models
        st.divider()
        st.subheader("Prediction Results")
        
        col_results = st.columns(3)
        
        for idx, (model_name, (model, _)) in enumerate(models.items()):
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            with col_results[idx]:
                st.markdown(f"### {model_name}")
                if prediction == 1:
                    st.success(f"✅ APPROVED")
                    st.metric("Approval Probability", f"{probability[1]*100:.1f}%")
                else:
                    st.error(f"❌ REJECTED")
                    st.metric("Rejection Probability", f"{probability[0]*100:.1f}%")

elif page == "📊 Model Performance":
    st.header("Model Performance Metrics")
    
    # Display metrics for each model
    model_names = list(results.keys())
    
    col1, col2, col3 = st.columns(3)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for idx, model_name in enumerate(model_names):
        with st.container():
            st.subheader(f"{model_name}")
            
            metrics_col = st.columns(4)
            for m_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                with metrics_col[m_idx]:
                    st.metric(metric_name, f"{results[model_name][metric]:.3f}")
            
            # Confusion Matrix
            cm = results[model_name]['cm']
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {model_name}')
            st.pyplot(fig)
            st.divider()

elif page == "📈 Data Analysis":
    st.header("Dataset Analysis & Visualizations")
    
    # Load original data for visualization
    df_original = pd.read_csv("loan_approval_data.csv")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Distribution", "Correlation", "Categorical"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Shape")
            st.metric("Total Records", len(df_original))
            st.metric("Total Features", len(df_original.columns))
        
        with col2:
            st.subheader("Loan Approval Distribution")
            approval_counts = df_original['Loan_Approved'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(approval_counts, labels=['No', 'Yes'], autopct='%1.1f%%', colors=['#ff9999', '#90EE90'])
            ax.set_title("Loan Approval Distribution")
            st.pyplot(fig)
    
    with tab2:
        # Income distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df_original['Applicant_Income'].dropna(), bins=20, color='skyblue', edgecolor='black')
        ax.set_xlabel("Applicant Income")
        ax.set_ylabel("Frequency")
        ax.set_title("Applicant Income Distribution")
        st.pyplot(fig)
        
        # Credit Score distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df_original['Credit_Score'].dropna(), bins=20, color='lightcoral', edgecolor='black')
        ax.set_xlabel("Credit Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Credit Score Distribution")
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Feature Correlation with Loan Approval")
        df_temp = df_original.copy()
        # Convert Loan_Approved to numeric
        df_temp['Loan_Approved'] = (df_temp['Loan_Approved'] == 'Yes').astype(int)
        
        num_cols = df_temp.select_dtypes(include=['float64', 'int64']).columns
        correlation_with_approval = df_temp[num_cols].corr()['Loan_Approved'].sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        correlation_with_approval.drop('Loan_Approved').plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel("Correlation with Loan Approval")
        ax.set_title("Feature Correlation with Loan Approval")
        st.pyplot(fig)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gender Distribution")
            gender_counts = df_original['Gender'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(gender_counts.index, gender_counts.values, color=['#FF69B4', '#4169E1'])
            ax.set_title("Gender Distribution")
            st.pyplot(fig)
        
        with col2:
            st.subheader("Education Level Distribution")
            edu_counts = df_original['Education_Level'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(edu_counts.index, edu_counts.values, color=['#FFA500', '#32CD32'])
            ax.set_title("Education Level Distribution")
            st.pyplot(fig)

elif page == "ℹ️ About":
    st.header("About This Dashboard")
    
    st.markdown("""
    ### 🎯 Purpose
    This dashboard provides a comprehensive tool for predicting loan approvals using machine learning models trained on loan approval data.
    
    ### 📊 Models Used
    - **Logistic Regression**: A linear model for binary classification
    - **K-Nearest Neighbors (KNN)**: Non-parametric model using 7 nearest neighbors
    - **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
    
    ### 🔧 Features
    1. **Prediction**: Input applicant details and get instant predictions from all three models
    2. **Model Performance**: View detailed metrics and confusion matrices for each model
    3. **Data Analysis**: Explore visualizations of the training dataset and feature relationships
    
    ### 📈 Key Metrics
    - **Accuracy**: Proportion of correct predictions
    - **Precision**: Proportion of positive predictions that are correct
    - **Recall**: Proportion of actual positives correctly identified
    - **F1 Score**: Harmonic mean of precision and recall
    
    ### 💡 Tips for Better Predictions
    - Higher credit scores improve approval chances
    - Lower DTI (Debt-to-Income) ratio is favorable
    - Stable employment and higher savings increase chances
    - Consistent income history matters
    
    ### 📋 Dataset Information
    - **Total Records**: 682
    - **Features**: 20+ including income, credit score, employment, demographics
    - **Target**: Loan Approved (Yes/No)
    
    ---
    *Built with Streamlit, Scikit-learn, and Pandas*
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Credit Wise Loan Approval Dashboard | Developed with ❤️ using Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
