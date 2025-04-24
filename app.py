from flask import Flask, render_template, request, jsonify
from datetime import datetime
from analyzer import EnhancedMisinfoDetector

app = Flask(__name__)

# Load Imran’s model
detector = EnhancedMisinfoDetector()

# Imran's UI Page
@app.route('/templates')
def imran_page():
    return render_template('imran.html')


# Analyze Claim
@app.route('/analyze', methods=['POST'])
def analyze_claim():
    data = request.json
    claim_data = {
        "id": f"claim-{datetime.now().timestamp()}",
        "text": data['claim'],
        "source": data.get('source', 'unknown'),
        "timestamp": datetime.now()
    }
    results = detector.analyze_claim(claim_data)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
