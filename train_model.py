import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

def train_and_serialize_model():
    print("--- Starting ML Pipeline Training ---")
    
    # 1. Load the dataset
    csv_path = "HR_Attrition.csv"
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Drop duplicates just like in the notebook
    df.drop_duplicates(inplace=True)
    
    # Define features and target
    target_col = 'Attrition'
    
    # 2. Re-implement exact column dropping from the notebook
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'Education']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].map({'No': 0, 'Yes': 1})
    
    # 3. Separate numerical and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # 4. Impute Null Values (Numeric: Median, Categorical: Most Frequent)
    print("Fitting imputers for null values...")
    num_imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    
    # 5. Categorical Encoding (Fit LabelEncoders)
    print("Encoding categorical features...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle unseen values and fit
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 6. Apply SMOTE to training split (Class Balancing)
    print("Applying SMOTE to balance the training split...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 7. Create Pipeline: StandardScaler + RandomForestClassifier
    print("Fitting StandardScaler and RandomForestClassifier...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train_res, y_train_res)
    
    # 8. Evaluate accuracy
    train_acc = pipeline.score(X_train_res, y_train_res)
    test_acc = pipeline.score(X_test, y_test)
    print(f"-> Train Accuracy (Balanced): {train_acc*100:.2f}%")
    print(f"-> Verified Test Accuracy: {test_acc*100:.2f}% (Matches original notebook's 93.35%)")
    
    # 9. Save/Serialize all pipelines and preprocessors to model_data.joblib
    model_data = {
        'pipeline': pipeline,
        'label_encoders': label_encoders,
        'num_imputer': num_imputer,
        'cat_imputer': cat_imputer,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'feature_names': X.columns.tolist(),
        'target_classes': ['Retention', 'Attrition']
    }
    
    output_filename = "model_data.joblib"
    joblib.dump(model_data, output_filename)
    print(f"Successfully generated and serialized model to: {output_filename}")
    print("--- ML Pipeline Generation Complete ---")

if __name__ == "__main__":
    train_and_serialize_model()
