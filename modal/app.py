from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
xgb_model = joblib.load('xgb_model.pkl')

# Mapping dictionaries for categorical features (define based on your model training)
store_type_mapping = {'S1': 1, 'S2': 2, 'S3': 3}  # Example mappings; replace with actual
location_type_mapping = {'L1': 1, 'L2': 2, 'L3': 3}
region_code_mapping = {'R1': 1, 'R2': 2, 'R3': 3}
week_day_mapping = {
    'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
    'Friday': 5, 'Saturday': 6, 'Sunday': 7
}
discount_mapping = {'Yes': 1, 'No': 0}

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions.
    Expects a JSON payload with input features.
    """
    try:
        # Get the JSON payload
        data = request.get_json()
        
        # Convert input data to numeric values as required by the model
        input_data = np.array([
            int(data['Store_id']),
            store_type_mapping.get(data['Store_Type'], 0),
            location_type_mapping.get(data['Location_Type'], 0),
            region_code_mapping.get(data['Region_Code'], 0),
            int(data['Holiday']),
            discount_mapping.get(data['Discount'], 0),
            int(data['Year']),
            int(data['Month']),
            int(data['Day']),
            week_day_mapping.get(data['Week Day'], 0),
            int(data['Holidays in Year']),
            int(data['Holidays in Month']),
            int(data['Dicounts in Year']),
            int(data['Dicounts in Month'])
        ]).reshape(1, -1)  # Reshape to a single sample

        # Make prediction
        prediction = xgb_model.predict(input_data)[0]

        # Convert prediction to a standard float for JSON serialization
        prediction = float(prediction)

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)


#     {

#     "Store_id":"1",
#     "Store_Type":"S1",
#     "Location_Type":"L3",
#     "Region_Code":"R1",
#     "Holiday":"1",
#     "Discount":"Yes",
#     "Year":"2018",
#     "Month":"01",
#     "Day":"01",
#     "Week Day":"Monday",
#     "Holidays in Year":"17155",
#     "Holidays in Month":"3650",
#     "Dicounts in Year":"60904",
#     "Dicounts in Month":"6227"    

# }