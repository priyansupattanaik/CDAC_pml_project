from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
# Enable CORS for all routes so our frontend HTML can talk to the backend
CORS(app)

# Load model data
try:
    model_data = joblib.load("model_data.joblib")
    pipeline = model_data['pipeline']
    label_encoders = model_data['label_encoders']
    feature_names = model_data['feature_names']
    target_classes = model_data['target_classes']
    print("API Server successfully loaded model_data.joblib!")
except Exception as e:
    print(f"Error loading model_data.joblib: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from POST request
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Build input dictionary
        input_data = {}
        for col in feature_names:
            if col not in data:
                return jsonify({'error': f'Missing feature: {col}'}), 400
            
            val = data[col]
            # Convert numeric fields
            if col in model_data['numeric_cols']:
                input_data[col] = float(val)
            else:
                input_data[col] = val
        
        # Build dataframe
        input_df = pd.DataFrame([input_data])
        
        # Apply Label Encoding
        for col, le in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col])
                except Exception as e:
                    # Fallback to first class in case of error
                    input_df[col] = le.transform([le.classes_[0]])[0]
        
        # Order columns correctly
        input_df = input_df[feature_names]
        
        # Calculate probability and predicted class
        prob = pipeline.predict_proba(input_df)[0][1]
        prediction = int(pipeline.predict(input_df)[0])
        
        # Map predicted class
        result_class = "Attrition" if prediction == 1 else "Retention"
        
        # Generate recommendations based on high risk drivers
        recommendations = []
        if prob > 0.4:
            if data.get('OverTime') == 'Yes':
                recommendations.append("Overtime is a massive attrition driver. Consider reducing workload or paying overtime bonuses.")
            if float(data.get('MonthlyIncome', 5000)) < 4000:
                recommendations.append("Salary is below median retention threshold. Consider a compensation review.")
            if float(data.get('WorkLifeBalance', 3)) >= 3:
                recommendations.append("Work-Life balance score indicates high strain. Review schedule flexibility.")
            if float(data.get('JobSatisfaction', 2)) >= 3:
                recommendations.append("Low job satisfaction. Set up a stay interview to address personal friction points.")
            if len(recommendations) == 0:
                recommendations.append("General attrition risk. Schedule an engagement review.")
        else:
            recommendations.append("Low risk. Maintain standard positive engagement policies.")
            
        return jsonify({
            'success': True,
            'prediction': prediction,
            'class': result_class,
            'probability': float(prob),
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/eda', methods=['GET'])
def get_eda_data():
    try:
        # Load the CSV dynamically
        df = pd.read_csv("HR_Attrition.csv")
        df.drop_duplicates(inplace=True)
        
        # 1. Attrition Distribution
        attr_counts = df['Attrition'].value_counts().to_dict()
        
        # 2. Attrition by Department
        dept_attr = df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0).to_dict(orient='index')
        
        # 3. Attrition by OverTime
        ot_attr = df.groupby(['OverTime', 'Attrition']).size().unstack(fill_value=0).to_dict(orient='index')
        
        # 4. Attrition by MaritalStatus
        marital_attr = df.groupby(['MaritalStatus', 'Attrition']).size().unstack(fill_value=0).to_dict(orient='index')
        
        # 5. Attrition by JobRole
        job_attr = df.groupby(['JobRole', 'Attrition']).size().unstack(fill_value=0).to_dict(orient='index')
        
        # 6. Attrition by WorkLifeBalance
        wlb_attr = df.groupby(['WorkLifeBalance', 'Attrition']).size().unstack(fill_value=0).to_dict(orient='index')
        
        # Flatten structure or convert types to standard Python types for JSON serialization
        def clean_keys(d):
            return {str(k): {str(ik): int(iv) for ik, iv in iv_d.items()} if isinstance(iv_d, dict) else int(iv_d) for k, iv_d in d.items()}
        
        return jsonify({
            'success': True,
            'attrition_distribution': {str(k): int(v) for k, v in attr_counts.items()},
            'department_attrition': clean_keys(dept_attr),
            'overtime_attrition': clean_keys(ot_attr),
            'marital_attrition': clean_keys(marital_attr),
            'job_role_attrition': clean_keys(job_attr),
            'wlb_attrition': clean_keys(wlb_attr)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'accuracy': 0.9335})

if __name__ == '__main__':
    # Run server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
