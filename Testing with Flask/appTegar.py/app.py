from flask import Flask, request, jsonify
from utils import load_tokenizer, load_model_file, predict_text

app = Flask(__name__)

# Memuat tokenizer dan model
tokenizer = load_tokenizer('ninja.pkl')
model = load_model_file('ninja.keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Memeriksa apakah input JSON valid
        if not request.is_json:
            return jsonify({'error': 'Invalid input, JSON data required'}), 400

        # Mengambil data dari request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Invalid input, "text" key is required'}), 400

        # Validasi teks
        text = data['text']
        if not isinstance(text, str) or text.strip() == '':
            return jsonify({'error': 'Invalid input, "text" must be a non-empty string'}), 400

        # Prediksi
        predicted_class, predicted_prob = predict_text(model, tokenizer, text)

        # Respon sukses
        response = {
            'prediction': predicted_class,
            'confidence': f"{predicted_prob:.2f}%"
        }
        return jsonify(response), 200

    except Exception as e:
        # Respon kesalahan internal
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method Not Allowed'}), 405


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not Found'}), 404


if __name__ == '__main__':
    app.run(debug=True)
