from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)   

with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_price():
    if request.method == 'POST':
        City = request.form.get('City')
        Type = request.form.get('type')
        room_number = request.form.get('room_number')
        room_number= float(room_number)
        Area = request.form.get('Area')
        Area = float(Area)   
        hasParking = request.form.get('hasParking')
        hasMamad= request.form.get('hasMamad')
        hasElevator= request.form.get('hasElevator')

        data = pd.DataFrame({
            'City': [City],
            'type': [Type],
            'room_number': [room_number],
            'Area': [Area],
            'hasParking': [hasParking],
            'hasMamad': [hasMamad],
            'hasElevator': [hasElevator]
        })
         
        Citys = ['אילת', 'באר שבע', 'בית שאן', 'בת ים', 'גבעת שמואל', 'דימונה', 'הוד השרון', 'הרצליה', 'זכרון יעקב', 'חולון', 'חיפה', 'יהוד מונוסון', 'ירושלים', 'כפר סבא', 'מודיעין מכבים רעות', 'נהריה', 'נוף הגליל', 'נס ציונה', 'נתניה', 'פתח תקווה', 'צפת', 'קרית ביאליק', 'ראשון לציון', 'רחובות', 'רמת גן', 'רעננה', 'שוהם', 'תל אביב', 'בית פרטי', 'דו משפחתי', 'דירה בבניין', 'דירת גן', 'פנטהאוז']
        for feature in Citys:
            data[feature] = 0
            
        encoded_data = preprocessor.transform(data)
        predicted_price = model.predict(encoded_data)[0]

        return render_template('index.html', predicted_price=predicted_price)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
