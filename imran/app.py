from flask import Flask, render_template, request, jsonify
from analyzer import EnhancedMisinfoDetector
from datetime import datetime, timedelta
import json

app = Flask(__name__)
detector = MisDisInfoDetector()


KNOWN_FACTS = [
    "Clinical trials show the vaccine has mild side effects in less than 10% of recipients",
    "The vaccine has been approved by major health organizations"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    claim_data = {
        "text": data['claim'],
        "source": data.get('source', 'unknown'),
        "timestamp": datetime.now(),
        "context": data.get('context', '')
    }
    
    results = detector.analyze_claim(claim_data, known_facts=KNOWN_FACTS)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
