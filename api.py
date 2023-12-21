# api.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(name)
model = load_model('group_9.h5')  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        file = request.files['file']
        img = image.load_img(file, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  

        
        predictions = model.predict(img_array)

        
        predicted_class = np.argmax(predictions[0])

        return jsonify({'class': predicted_class, 'confidence': float(predictions[0][predicted_class])})
    except Exception as e:
        return jsonify({'error': str(e)})

if name == 'main':
    app.run(debug=True, host='0.0.0.0', port=3300)