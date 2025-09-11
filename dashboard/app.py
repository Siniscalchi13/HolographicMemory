from flask import Flask, render_template, jsonify, send_from_directory
import json
import time
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'orchestrator': 'healthy',
            'router': 'healthy',
            'vault': 'healthy',
            'telemetry': 'healthy'
        },
        'gpu': {
            'available': True,
            'usage': 30
        },
        'memory': {
            'used': 2.1,
            'total': 8.0,
            'percentage': 65
        }
    })

@app.route('/api/metrics')
def api_metrics():
    return jsonify({
        'cpu_usage': 45,
        'memory_usage': 65,
        'gpu_usage': 30,
        'network_io': 20,
        'compression_ratio': 1475.0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/assets/<path:filename>')
def assets(filename: str):
    """Serve SmartHaus assets from static/assets"""
    return send_from_directory('static/assets', filename)

if __name__ == '__main__':
    import sys
    port = 5000
    if len(sys.argv) > 1 and sys.argv[1] == '--port' and len(sys.argv) > 2:
        try:
            port = int(sys.argv[2])
        except ValueError:
            print(f"Invalid port: {sys.argv[2]}, using default 5000")
    app.run(debug=True, host='0.0.0.0', port=port)
