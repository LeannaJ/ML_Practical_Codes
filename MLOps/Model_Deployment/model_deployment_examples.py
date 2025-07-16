"""
Model Deployment Examples
=========================

- Flask API for Model Serving
- FastAPI for High-Performance Serving
- Docker Containerization
- Model Versioning and A/B Testing
- Load Balancing and Scaling
- Health Checks and Monitoring
- Deployment Best Practices
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import time
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Web framework imports
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn")

print("=== Model Deployment Examples ===")

# 1. Train and Save Model
print("\n=== Training and Saving Model ===")

def train_model():
    """Train a simple model for deployment"""
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                             n_redundant=3, n_repeated=2, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    return pipeline, X_test, y_test

# Train model
model, X_test, y_test = train_model()

# Save model
model_path = "deployment_model.joblib"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# 2. Flask API for Model Serving
print("\n=== Flask API for Model Serving ===")

if FLASK_AVAILABLE:
    app = Flask(__name__)
    
    # Load model
    loaded_model = joblib.load(model_path)
    
    # Model metadata
    model_metadata = {
        'model_name': 'Random Forest Classifier',
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'features': 20,
        'classes': 2
    }
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': loaded_model is not None
        })
    
    @app.route('/model/info', methods=['GET'])
    def model_info():
        """Get model information"""
        return jsonify(model_metadata)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Make predictions"""
        try:
            # Get input data
            data = request.get_json()
            
            if not data or 'features' not in data:
                return jsonify({'error': 'No features provided'}), 400
            
            # Convert to numpy array
            features = np.array(data['features'])
            
            # Validate input
            if features.shape[0] != model_metadata['features']:
                return jsonify({
                    'error': f'Expected {model_metadata["features"]} features, got {features.shape[0]}'
                }), 400
            
            # Make prediction
            prediction = loaded_model.predict([features])[0]
            probability = loaded_model.predict_proba([features])[0].tolist()
            
            return jsonify({
                'prediction': int(prediction),
                'probabilities': probability,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/predict/batch', methods=['POST'])
    def predict_batch():
        """Make batch predictions"""
        try:
            # Get input data
            data = request.get_json()
            
            if not data or 'features' not in data:
                return jsonify({'error': 'No features provided'}), 400
            
            # Convert to numpy array
            features = np.array(data['features'])
            
            # Validate input
            if features.shape[1] != model_metadata['features']:
                return jsonify({
                    'error': f'Expected {model_metadata["features"]} features, got {features.shape[1]}'
                }), 400
            
            # Make predictions
            predictions = loaded_model.predict(features).tolist()
            probabilities = loaded_model.predict_proba(features).tolist()
            
            return jsonify({
                'predictions': predictions,
                'probabilities': probabilities,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def run_flask_app():
        """Run Flask app"""
        print("Starting Flask API server...")
        print("API endpoints:")
        print("  GET  /health - Health check")
        print("  GET  /model/info - Model information")
        print("  POST /predict - Single prediction")
        print("  POST /predict/batch - Batch prediction")
        
        # Run in development mode
        app.run(host='0.0.0.0', port=5000, debug=False)

# 3. FastAPI for High-Performance Serving
print("\n=== FastAPI for High-Performance Serving ===")

if FASTAPI_AVAILABLE:
    fastapi_app = FastAPI(
        title="ML Model API",
        description="High-performance API for machine learning model serving",
        version="1.0.0"
    )
    
    # Load model
    fastapi_model = joblib.load(model_path)
    
    @fastapi_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": fastapi_model is not None
        }
    
    @fastapi_app.get("/model/info")
    async def model_info():
        """Get model information"""
        return model_metadata
    
    @fastapi_app.post("/predict")
    async def predict(features: list):
        """Make single prediction"""
        try:
            # Convert to numpy array
            features_array = np.array(features)
            
            # Validate input
            if features_array.shape[0] != model_metadata['features']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {model_metadata['features']} features, got {features_array.shape[0]}"
                )
            
            # Make prediction
            prediction = fastapi_model.predict([features_array])[0]
            probability = fastapi_model.predict_proba([features_array])[0].tolist()
            
            return {
                "prediction": int(prediction),
                "probabilities": probability,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @fastapi_app.post("/predict/batch")
    async def predict_batch(features: list):
        """Make batch predictions"""
        try:
            # Convert to numpy array
            features_array = np.array(features)
            
            # Validate input
            if features_array.shape[1] != model_metadata['features']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {model_metadata['features']} features, got {features_array.shape[1]}"
                )
            
            # Make predictions
            predictions = fastapi_model.predict(features_array).tolist()
            probabilities = fastapi_model.predict_proba(features_array).tolist()
            
            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def run_fastapi_app():
        """Run FastAPI app"""
        print("Starting FastAPI server...")
        print("API endpoints:")
        print("  GET  /health - Health check")
        print("  GET  /model/info - Model information")
        print("  POST /predict - Single prediction")
        print("  POST /predict/batch - Batch prediction")
        print("  GET  /docs - API documentation")
        
        # Run with uvicorn
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

# 4. Model Versioning and A/B Testing
print("\n=== Model Versioning and A/B Testing ===")

class ModelVersionManager:
    """Model version manager for A/B testing"""
    
    def __init__(self, base_path="model_versions"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.versions = {}
        self.active_versions = {}
        self.load_versions()
    
    def load_versions(self):
        """Load version information"""
        version_file = os.path.join(self.base_path, "versions.json")
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                self.versions = json.load(f)
    
    def save_versions(self):
        """Save version information"""
        version_file = os.path.join(self.base_path, "versions.json")
        with open(version_file, 'w') as f:
            json.dump(self.versions, f, indent=4)
    
    def save_model_version(self, model, version, description="", traffic_split=0.0):
        """Save a new model version"""
        version_path = os.path.join(self.base_path, f"model_v{version}")
        os.makedirs(version_path, exist_ok=True)
        
        # Save model
        model_file = os.path.join(version_path, "model.joblib")
        joblib.dump(model, model_file)
        
        # Save metadata
        metadata = {
            'version': version,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'model_file': model_file,
            'traffic_split': traffic_split,
            'active': False
        }
        
        metadata_file = os.path.join(version_path, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Update versions
        self.versions[version] = metadata
        self.save_versions()
        
        print(f"Model version {version} saved successfully")
    
    def activate_version(self, version, traffic_split=1.0):
        """Activate a model version"""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        # Load model
        model_file = self.versions[version]['model_file']
        model = joblib.load(model_file)
        
        # Update metadata
        self.versions[version]['active'] = True
        self.versions[version]['traffic_split'] = traffic_split
        self.versions[version]['activated_at'] = datetime.now().isoformat()
        
        # Store in active versions
        self.active_versions[version] = model
        
        self.save_versions()
        print(f"Model version {version} activated with {traffic_split*100}% traffic")
    
    def deactivate_version(self, version):
        """Deactivate a model version"""
        if version in self.active_versions:
            del self.active_versions[version]
            self.versions[version]['active'] = False
            self.save_versions()
            print(f"Model version {version} deactivated")
    
    def predict_with_ab_testing(self, features):
        """Make prediction with A/B testing"""
        if not self.active_versions:
            raise ValueError("No active model versions")
        
        # Simple A/B testing: randomly select version based on traffic split
        import random
        
        # Get active versions and their traffic splits
        active_versions = [(v, self.versions[v]['traffic_split']) 
                          for v in self.active_versions.keys()]
        
        # Normalize traffic splits
        total_traffic = sum(split for _, split in active_versions)
        if total_traffic == 0:
            # If no traffic split defined, use equal distribution
            active_versions = [(v, 1.0/len(active_versions)) 
                              for v in self.active_versions.keys()]
            total_traffic = 1.0
        
        # Select version based on traffic split
        rand = random.random() * total_traffic
        cumulative = 0
        
        for version, split in active_versions:
            cumulative += split
            if rand <= cumulative:
                # Use this version
                model = self.active_versions[version]
                prediction = model.predict([features])[0]
                probability = model.predict_proba([features])[0].tolist()
                
                return {
                    'prediction': int(prediction),
                    'probabilities': probability,
                    'model_version': version,
                    'timestamp': datetime.now().isoformat()
                }
        
        # Fallback to first active version
        version = list(self.active_versions.keys())[0]
        model = self.active_versions[version]
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0].tolist()
        
        return {
            'prediction': int(prediction),
            'probabilities': probability,
            'model_version': version,
            'timestamp': datetime.now().isoformat()
        }

# Create model version manager
version_manager = ModelVersionManager()

# Save different model versions
version_manager.save_model_version(model, "1.0", "Initial model")
version_manager.save_model_version(model, "2.0", "Updated model")

# Activate versions for A/B testing
version_manager.activate_version("1.0", traffic_split=0.7)
version_manager.activate_version("2.0", traffic_split=0.3)

# Test A/B testing
test_features = X_test[0]
ab_result = version_manager.predict_with_ab_testing(test_features)
print(f"A/B Testing Result: {ab_result}")

# 5. Load Balancing and Scaling
print("\n=== Load Balancing and Scaling ===")

class LoadBalancer:
    """Simple load balancer for model serving"""
    
    def __init__(self, models):
        self.models = models
        self.current_index = 0
        self.request_counts = {i: 0 for i in range(len(models))}
    
    def round_robin(self, features):
        """Round-robin load balancing"""
        model = self.models[self.current_index]
        self.request_counts[self.current_index] += 1
        self.current_index = (self.current_index + 1) % len(self.models)
        
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0].tolist()
        
        return {
            'prediction': int(prediction),
            'probabilities': probability,
            'server_id': self.current_index,
            'timestamp': datetime.now().isoformat()
        }
    
    def least_connections(self, features):
        """Least connections load balancing"""
        # Find server with least requests
        min_requests = min(self.request_counts.values())
        candidates = [i for i, count in self.request_counts.items() 
                     if count == min_requests]
        
        # Select random candidate
        import random
        selected_index = random.choice(candidates)
        model = self.models[selected_index]
        self.request_counts[selected_index] += 1
        
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0].tolist()
        
        return {
            'prediction': int(prediction),
            'probabilities': probability,
            'server_id': selected_index,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_stats(self):
        """Get load balancer statistics"""
        return {
            'request_counts': self.request_counts,
            'total_requests': sum(self.request_counts.values()),
            'average_requests': sum(self.request_counts.values()) / len(self.models)
        }

# Create multiple model instances for load balancing
models_for_lb = [joblib.load(model_path) for _ in range(3)]
load_balancer = LoadBalancer(models_for_lb)

# Test load balancing
for i in range(10):
    result = load_balancer.round_robin(X_test[i])
    print(f"Request {i+1}: Server {result['server_id']}, Prediction: {result['prediction']}")

print(f"Load Balancer Stats: {load_balancer.get_stats()}")

# 6. Health Checks and Monitoring
print("\n=== Health Checks and Monitoring ===")

class ModelMonitor:
    """Model monitoring and health checks"""
    
    def __init__(self, model):
        self.model = model
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.predictions = []
        self.start_time = datetime.now()
    
    def health_check(self):
        """Perform health check"""
        try:
            # Test with dummy data
            dummy_features = np.random.randn(20)
            start_time = time.time()
            
            prediction = self.model.predict([dummy_features])
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time': response_time,
                'model_loaded': self.model is not None,
                'uptime': (datetime.now() - self.start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_with_monitoring(self, features):
        """Make prediction with monitoring"""
        start_time = time.time()
        
        try:
            prediction = self.model.predict([features])[0]
            probability = self.model.predict_proba([features])[0].tolist()
            
            response_time = time.time() - start_time
            self.request_count += 1
            self.response_times.append(response_time)
            self.predictions.append(int(prediction))
            
            return {
                'prediction': int(prediction),
                'probabilities': probability,
                'response_time': response_time,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.error_count += 1
            raise e
    
    def get_metrics(self):
        """Get monitoring metrics"""
        if not self.response_times:
            return {}
        
        return {
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / self.request_count if self.request_count > 0 else 0,
            'average_response_time': np.mean(self.response_times),
            'min_response_time': np.min(self.response_times),
            'max_response_time': np.max(self.response_times),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'requests_per_second': self.request_count / ((datetime.now() - self.start_time).total_seconds()) if (datetime.now() - self.start_time).total_seconds() > 0 else 0
        }

# Create model monitor
monitor = ModelMonitor(model)

# Test monitoring
for i in range(5):
    try:
        result = monitor.predict_with_monitoring(X_test[i])
        print(f"Prediction {i+1}: {result['prediction']}, Response time: {result['response_time']:.4f}s")
    except Exception as e:
        print(f"Error in prediction {i+1}: {e}")

print(f"Health Check: {monitor.health_check()}")
print(f"Monitoring Metrics: {monitor.get_metrics()}")

# 7. Docker Configuration
print("\n=== Docker Configuration ===")

# Create Dockerfile
dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application
COPY deployment_model.joblib .
COPY app.py .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
"""

# Create requirements.txt
requirements_content = """
flask==2.0.1
scikit-learn==1.0.2
numpy==1.21.2
pandas==1.3.3
joblib==1.1.0
"""

# Create app.py for Docker
app_py_content = '''
import joblib
from flask import Flask, request, jsonify
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load model
model = joblib.load('deployment_model.joblib')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features'])
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0].tolist()
        
        return jsonify({
            'prediction': int(prediction),
            'probabilities': probability,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
'''

# Save Docker files
with open('Dockerfile', 'w') as f:
    f.write(dockerfile_content)

with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

with open('app.py', 'w') as f:
    f.write(app_py_content)

print("Docker configuration files created:")
print("- Dockerfile")
print("- requirements.txt")
print("- app.py")

# 8. Deployment Testing
print("\n=== Deployment Testing ===")

def test_api_endpoints():
    """Test API endpoints"""
    if not FLASK_AVAILABLE:
        print("Flask not available for testing")
        return
    
    # Start Flask app in a separate thread for testing
    import threading
    import time
    
    def run_app():
        app.run(host='0.0.0.0', port=5000, debug=False)
    
    # Start server
    server_thread = threading.Thread(target=run_app)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    try:
        # Test health check
        response = requests.get('http://localhost:5000/health')
        print(f"Health Check: {response.status_code} - {response.json()}")
        
        # Test model info
        response = requests.get('http://localhost:5000/model/info')
        print(f"Model Info: {response.status_code} - {response.json()}")
        
        # Test prediction
        test_data = {'features': X_test[0].tolist()}
        response = requests.post('http://localhost:5000/predict', json=test_data)
        print(f"Prediction: {response.status_code} - {response.json()}")
        
        # Test batch prediction
        test_batch_data = {'features': X_test[:3].tolist()}
        response = requests.post('http://localhost:5000/predict/batch', json=test_batch_data)
        print(f"Batch Prediction: {response.status_code} - {response.json()}")
        
    except Exception as e:
        print(f"Testing error: {e}")

# Run tests
test_api_endpoints()

# 9. Summary and Best Practices
print("\n=== Summary and Best Practices ===")

print("Deployment Components Created:")
print("=" * 50)
print("1. Flask API for model serving")
print("2. FastAPI for high-performance serving")
print("3. Model versioning and A/B testing")
print("4. Load balancing and scaling")
print("5. Health checks and monitoring")
print("6. Docker containerization")
print("7. API testing and validation")

print("\nDeployment Best Practices:")
print("=" * 50)
print("1. Always implement health checks for your APIs")
print("2. Use proper error handling and logging")
print("3. Implement model versioning for A/B testing")
print("4. Use load balancing for high availability")
print("5. Monitor model performance and response times")
print("6. Containerize your applications with Docker")
print("7. Use environment variables for configuration")
print("8. Implement proper API documentation")
print("9. Use HTTPS in production")
print("10. Set up automated testing and CI/CD")

print("\nDeployment Checklist:")
print("=" * 50)
print("□ Model serialization and loading")
print("□ API endpoint implementation")
print("□ Input validation and error handling")
print("□ Health check endpoints")
print("□ Model versioning system")
print("□ Load balancing configuration")
print("□ Monitoring and logging setup")
print("□ Docker containerization")
print("□ Environment configuration")
print("□ Security measures")
print("□ Performance testing")
print("□ Documentation")

print("\nNext Steps:")
print("=" * 50)
print("1. Deploy to cloud platform (AWS, GCP, Azure)")
print("2. Set up CI/CD pipeline")
print("3. Implement model monitoring dashboard")
print("4. Set up alerting for model performance")
print("5. Implement model retraining pipeline")
print("6. Set up data drift detection")
print("7. Implement model explainability")
print("8. Set up backup and disaster recovery") 