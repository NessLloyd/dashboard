from flask import Flask, render_template, request, jsonify
from datetime import datetime
from templates.imran.analyzer import EnhancedMisinfoDetector

app = Flask(__name__)

# Load Imran’s model
detector = EnhancedMisinfoDetector()

# Home Dashboard Page
@app.route('/')
def dashboard():
    return render_template('index.html')  # Your main dashboard

# Imran's UI Page
@app.route('/imran')
def imran_page():
    return render_template('imran/index.html')


# Analyze Claim
@app.route('/imran/analyze', methods=['POST'])
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
