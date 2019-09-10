from flask import Flask, render_template, request, jsonify
from predict import predict_sentence

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    text = request.json.get('text')
    sentences = text.split('\n')
    sentences = [s.strip() for s in sentences]
    sentences = [predict_sentence(s) for s in sentences]
    return jsonify(sentences)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
