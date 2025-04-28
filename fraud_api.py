from flask import Flask, request, jsonify
import pandas as pd
import pickle
from datetime import datetime
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# Load trained model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get expected feature names from the model
trained_features = model.get_booster().feature_names

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        input_data = request.json['features']
        
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Map admission/discharge dates to claim dates
        input_df['ClaimStartDt'] = input_df.get('AdmissionDt', '')
        input_df['ClaimEndDt'] = input_df.get('DischargeDt', '')
        
        # Convert numerical columns to float
        numerical_cols = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid']
        input_df[numerical_cols] = input_df[numerical_cols].astype(float)
        
        # Handle date columns
        date_columns = ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt']
        for col in date_columns:
            # Convert to datetime, handle missing/invalid dates
            input_df[col] = pd.to_datetime(input_df[col], errors='coerce')
            # Extract date components
            input_df[f'{col}_Day'] = input_df[col].dt.day.fillna(0)
            input_df[f'{col}_Month'] = input_df[col].dt.month.fillna(0)
            input_df[f'{col}_Year'] = input_df[col].dt.year.fillna(0)
        
        # Drop original date columns
        input_df.drop(columns=date_columns, inplace=True, errors='ignore')
        
        # Handle categorical encoding
        categorical_cols = ['BeneID', 'ClaimID', 'Provider', 'AttendingPhysician',
                            'OperatingPhysician', 'OtherPhysician']
        for col in categorical_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype('category').cat.codes
            else:
                input_df[col] = 0  # Default for missing categoricals
                
        # Ensure numerical columns exist
        numerical_cols = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid']
        for col in numerical_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Drop columns not used in training
        columns_to_drop = [
            'ClmAdmitDiagnosisCode', 'DiagnosisGroupCode',
            'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
            'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
            'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
            'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
            'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5',
            'ClmProcedureCode_6'
        ]
        input_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
        
        # Align features with model expectations
        input_df = input_df.reindex(columns=trained_features, fill_value=0)
        
        # Validate feature count
        if len(input_df.columns) != 20:
            return jsonify({'error': f'Feature mismatch : expected 20 features, got {len(input_df.columns)}'}), 400
        
        proba = model.predict_proba(input_df)[0][1]
        print("Fraud probability:", proba)
        return jsonify({
    'prediction': 'Fraud' if proba > 0.6 else 'Not Fraud',
    'probability': float(proba)  
})    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
   
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)