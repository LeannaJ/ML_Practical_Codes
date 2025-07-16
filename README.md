# ML Practical Codes - Complete Machine Learning Project Collection

A comprehensive repository containing practical Python implementations of machine learning algorithms, deep learning models, and MLOps solutions. This project covers the entire ML lifecycle from data preprocessing to production deployment.

## 📋 Table of Contents

- [📁 Overall Project Structure](#overall-project-structure)
- [🚀 Quick Start](#quick-start)
- [📁 Project Structure](#project-structure)
  - [🎯 Basic Machine Learning](#basic-machine-learning)
  - [🧠 Deep Learning](#deep-learning)
  - [📝 NLP](#nlp)
  - [🔧 Optimization](#optimization)
  - [📊 Predictive Modeling](#predictive-modeling)
  - [🎯 Clustering](#clustering)
  - [👁️ Computer Vision](#computer-vision)
  - [🚀 MLOps](#mlops)
- [🛠️ Technologies & Libraries](#technologies-libraries)
- [📚 Learning Path](#learning-path)
- [🎯 Use Cases](#use-cases)
- [🔧 Customization](#customization)
- [📝 Contributing](#contributing)
- [📄 License](#license)
- [👤 Author](#author)

## 📁 Overall Project Structure

<a name="overall-project-structure"></a>

```
ML_Practical_Codes/
├── 📊 Basic_Machine_Learning/
│   ├── 📈 Advanced_Ensemble_Methods/
│   │   ├── xgboost_examples.py
│   │   ├── lightgbm_examples.py
│   │   ├── catboost_examples.py
│   │   ├── random_forest_examples.py
│   │   └── ensemble_comparison.py
│   ├── 🧹 Data_Preprocessing/
│   │   └── data_preprocessing_basic.py
│   ├── 📊 EDA/
│   │   └── exploratory_data_analysis.py
│   ├── 🔧 Feature_Engineering/
│   │   └── feature_engineering_techniques.py
│   ├── 🎯 Model_Evaluation/
│   │   └── model_evaluation_comprehensive.py
│   ├── 🚀 Model_Training/
│   │   └── model_training_pipeline.py
│   ├── 📈 Regression/
│   │   ├── regression_algorithms_comparison.py
│   │   ├── business_regression_examples.py
│   │   └── regression_boston.ipynb
│   └── 🛠️ Utilities/
│       └── ml_utilities.py
├── 🧠 Deep_Learning/
│   ├── 🔍 Attention/
│   │   └── attention_mechanisms.py
│   ├── 🖼️ CNN/
│   │   ├── cnn_image_classification.py
│   │   ├── cnn_cifar10.ipynb
│   │   └── cnn_mnist.ipynb
│   ├── 🧠 DNN/
│   │   ├── deep_neural_networks.py
│   │   ├── dnn_mnist_trial1.ipynb
│   │   └── dnn_mnist_trial2.ipynb
│   ├── 🔄 LSTM/
│   │   ├── lstm_sequence_models.py
│   │   ├── lstm_sentiment.ipynb
│   │   └── lstm_text_gen.ipynb
│   ├── 🔄 RNN/
│   │   └── rnn_sequence_prediction.py
│   └── 🎨 VAE/
│       ├── variational_autoencoders.py
│       ├── vae_conv_mnist_HW_LeannaJeon.ipynb
│       └── vae_linear_mnist.ipynb
├── 📝 NLP/
│   ├── 🤖 Pretrained_Models/
│   │   └── pretrained_models_examples.py
│   ├── 🔍 Semantic_Analysis/
│   │   └── semantic_analysis_examples.py
│   ├── 📊 Text_Classification/
│   │   └── text_classification_examples.py
│   ├── 🧹 Text_Preprocessing/
│   │   └── text_preprocessing_examples.py
│   ├── 📚 Topic_Modeling/
│   │   └── topic_modeling_examples.py
│   └── 🔤 Word_Embeddings/
│       └── word_embeddings_examples.py
├── 🔧 Optimization/
│   ├── 📐 Convex_Optimization/
│   │   └── convex_optimization_examples.py
│   ├── 🧬 Genetic_Algorithms/
│   │   └── genetic_algorithm_examples.py
│   ├── 📊 Linear_Programming/
│   │   └── linear_programming_examples.py
│   ├── 📈 Nonlinear_Optimization/
│   │   └── nonlinear_optimization_examples.py
│   ├── 🐝 Particle_Swarm_Optimization/
│   │   └── particle_swarm_optimization_examples.py
│   └── ❄️ Simulated_Annealing/
│       └── simulated_annealing_examples.py
├── 📊 Predictive_Modeling/
│   ├── 🚨 Anomaly_Detection/
│   │   └── anomaly_detection_examples.py
│   ├── 🎯 Classification_Problems/
│   │   ├── classification_examples.py
│   │   ├── Credit_Risk_Prediction.ipynb
│   │   └── Insurance_Policy_Renewal_Prediction.ipynb
│   ├── 🏷️ Multi_Label_Classification/
│   │   └── multi_label_examples.py
│   ├── 💡 Recommendation_Systems/
│   │   ├── recommendation_examples.py
│   │   └── Segmentation_Recommendation_Online_Investment_Platforms.ipynb
│   ├── 📈 Regression_Problems/
│   │   ├── regression_examples.py
│   │   └── Predicting_South_Korean_Condominium_Prices.ipynb
│   └── ⏰ Time_Series_Forecasting/
│       ├── time_series_forecasting_examples.py
│       └── Bike_Rental_ Demand_Prediction.ipynb
├── 🎯 Clustering/
│   └── clustering_examples.py
├── 👁️ Computer_Vision/
│   ├── 👤 Face_Recognition/
│   │   └── face_recognition_examples.py
│   ├── 🔍 Feature_Extraction/
│   │   └── feature_extraction_examples.py
│   ├── 🖼️ Image_Classification/
│   │   └── image_classification_examples.py
│   ├── 🎨 Image_Segmentation/
│   │   └── image_segmentation_examples.py
│   └── 🎯 Object_Detection/
│       └── object_detection_examples.py
├── 🚀 MLOps/
│   ├── 🚀 Model_Deployment/
│   │   └── model_deployment_examples.py
│   ├── 🔄 Model_Pipeline/
│   │   └── ml_pipeline_examples.py
│   └── README.md
└── README.md
```

## 🚀 Quick Start

<a name="quick-start"></a>

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt

# Or install core dependencies
pip install numpy pandas scikit-learn matplotlib seaborn
pip install tensorflow torch transformers
pip install xgboost lightgbm catboost
pip install flask fastapi mlflow
pip install opencv-python pillow
pip install nltk spacy gensim
pip install scipy cvxpy pulp
```

### Running Examples
```bash
# Run any Python script directly
python Basic_Machine_Learning/Data_Preprocessing/data_preprocessing_basic.py

# Or run specific examples
python Deep_Learning/CNN/cnn_image_classification.py
python NLP/Text_Classification/text_classification_examples.py
python MLOps/Model_Deployment/model_deployment_examples.py
```

## 📁 Project Structure

<a name="project-structure"></a>

### 🎯 Basic Machine Learning

<a name="basic-machine-learning"></a>
Core ML fundamentals with practical implementations.

#### Data Preprocessing
- **`data_preprocessing_basic.py`** - Comprehensive data cleaning, handling missing values, outliers, and data transformation
- **Usage**: `python Basic_Machine_Learning/Data_Preprocessing/data_preprocessing_basic.py`

#### Exploratory Data Analysis (EDA)
- **`exploratory_data_analysis.py`** - Statistical analysis, data visualization, correlation analysis, and insights generation
- **Usage**: `python Basic_Machine_Learning/EDA/exploratory_data_analysis.py`

#### Feature Engineering
- **`feature_engineering_techniques.py`** - Feature selection, creation, scaling, encoding, and dimensionality reduction
- **Usage**: `python Basic_Machine_Learning/Feature_Engineering/feature_engineering_techniques.py`

#### Model Training
- **`model_training_pipeline.py`** - Complete ML pipeline with cross-validation, hyperparameter tuning, and model selection
- **Usage**: `python Basic_Machine_Learning/Model_Training/model_training_pipeline.py`

#### Model Evaluation
- **`model_evaluation_comprehensive.py`** - Performance metrics, confusion matrices, ROC curves, and model interpretation
- **Usage**: `python Basic_Machine_Learning/Model_Evaluation/model_evaluation_comprehensive.py`

#### Regression
- **`regression_algorithms_comparison.py`** - Linear, Ridge, Lasso, Elastic Net, SVR, and Random Forest regression
- **`business_regression_examples.py`** - Real-world business regression problems
- **Usage**: `python Basic_Machine_Learning/Regression/regression_algorithms_comparison.py`

#### Advanced Ensemble Methods
- **`xgboost_examples.py`** - XGBoost for classification and regression with hyperparameter tuning
- **`lightgbm_examples.py`** - LightGBM implementation with feature importance analysis
- **`catboost_examples.py`** - CatBoost with categorical feature handling
- **`random_forest_examples.py`** - Random Forest with ensemble techniques
- **`ensemble_comparison.py`** - Comprehensive comparison of all ensemble methods
- **Usage**: `python Basic_Machine_Learning/Advanced_Ensemble_Methods/ensemble_comparison.py`

#### Utilities
- **`ml_utilities.py`** - Helper functions for data loading, model saving, and common ML operations
- **Usage**: `python Basic_Machine_Learning/Utilities/ml_utilities.py`

### 🧠 Deep Learning

<a name="deep-learning"></a>
Neural network implementations using TensorFlow/Keras and PyTorch.

#### CNN (Convolutional Neural Networks)
- **`cnn_image_classification.py`** - CNN for image classification with data augmentation
- **`cnn_cifar10.ipynb`** - CIFAR-10 classification notebook
- **`cnn_mnist.ipynb`** - MNIST digit classification notebook
- **Usage**: `python Deep_Learning/CNN/cnn_image_classification.py`

#### DNN (Deep Neural Networks)
- **`deep_neural_networks.py`** - Multi-layer perceptrons with various architectures
- **`dnn_mnist_trial1.ipynb`** - DNN implementation notebook
- **`dnn_mnist_trial2.ipynb`** - Advanced DNN techniques notebook
- **Usage**: `python Deep_Learning/DNN/deep_neural_networks.py`

#### LSTM (Long Short-Term Memory)
- **`lstm_sequence_models.py`** - LSTM for time series prediction and sequence modeling
- **`lstm_sentiment.ipynb`** - Sentiment analysis notebook
- **`lstm_text_gen.ipynb`** - Text generation notebook
- **Usage**: `python Deep_Learning/LSTM/lstm_sequence_models.py`

#### RNN (Recurrent Neural Networks)
- **`rnn_sequence_prediction.py`** - RNN implementations for sequence prediction tasks
- **Usage**: `python Deep_Learning/RNN/rnn_sequence_prediction.py`

#### Attention Mechanisms
- **`attention_mechanisms.py`** - Self-attention, multi-head attention, and transformer components
- **Usage**: `python Deep_Learning/Attention/attention_mechanisms.py`

#### VAE (Variational Autoencoders)
- **`variational_autoencoders.py`** - VAE implementation for generative modeling
- **`vae_conv_mnist_HW_LeannaJeon.ipynb`** - Convolutional VAE notebook
- **`vae_linear_mnist.ipynb`** - Linear VAE notebook
- **Usage**: `python Deep_Learning/VAE/variational_autoencoders.py`

### 📝 NLP (Natural Language Processing)

<a name="nlp"></a>
Text processing and language understanding implementations.

#### Text Preprocessing
- **`text_preprocessing_examples.py`** - Tokenization, stemming, lemmatization, and text cleaning
- **Usage**: `python NLP/Text_Preprocessing/text_preprocessing_examples.py`

#### Word Embeddings
- **`word_embeddings_examples.py`** - Word2Vec, GloVe, FastText, and custom embeddings
- **Usage**: `python NLP/Word_Embeddings/word_embeddings_examples.py`

#### Text Classification
- **`text_classification_examples.py`** - Naive Bayes, SVM, LSTM, and BERT for text classification
- **Usage**: `python NLP/Text_Classification/text_classification_examples.py`

#### Topic Modeling
- **`topic_modeling_examples.py`** - LDA, NMF, and BERTopic for document clustering
- **Usage**: `python NLP/Topic_Modeling/topic_modeling_examples.py`

#### Pretrained Models
- **`pretrained_models_examples.py`** - BERT, GPT, and other transformer models
- **Usage**: `python NLP/Pretrained_Models/pretrained_models_examples.py`

#### Semantic Analysis
- **`semantic_analysis_examples.py`** - Sentiment analysis, text similarity, and semantic search
- **Usage**: `python NLP/Semantic_Analysis/semantic_analysis_examples.py`

### 🔧 Optimization

<a name="optimization"></a>
Mathematical optimization algorithms and techniques.

#### Linear Programming
- **`linear_programming_examples.py`** - MILP, transportation problems, and resource allocation
- **Usage**: `python Optimization/Linear_Programming/linear_programming_examples.py`

#### Nonlinear Optimization
- **`nonlinear_optimization_examples.py`** - Gradient descent variants and constraint optimization
- **Usage**: `python Optimization/Nonlinear_Optimization/nonlinear_optimization_examples.py`

#### Genetic Algorithms
- **`genetic_algorithm_examples.py`** - GA for function optimization and combinatorial problems
- **Usage**: `python Optimization/Genetic_Algorithms/genetic_algorithm_examples.py`

#### Simulated Annealing
- **`simulated_annealing_examples.py`** - SA for global optimization and scheduling problems
- **Usage**: `python Optimization/Simulated_Annealing/simulated_annealing_examples.py`

#### Particle Swarm Optimization
- **`particle_swarm_optimization_examples.py`** - PSO for continuous optimization problems
- **Usage**: `python Optimization/Particle_Swarm_Optimization/particle_swarm_optimization_examples.py`

#### Convex Optimization
- **`convex_optimization_examples.py`** - Convex problems, quadratic programming, and portfolio optimization
- **Usage**: `python Optimization/Convex_Optimization/convex_optimization_examples.py`

### 📊 Predictive Modeling

<a name="predictive-modeling"></a>
Real-world prediction problems and solutions.

#### Time Series Forecasting
- **`time_series_forecasting_examples.py`** - ARIMA, Prophet, LSTM, and ensemble methods
- **`Bike_Rental_ Demand_Prediction.ipynb`** - Bike rental demand prediction
- **Usage**: `python Predictive_Modeling/Time_Series_Forecasting/time_series_forecasting_examples.py`

#### Classification Problems
- **`classification_examples.py`** - Binary and multiclass classification with various algorithms
- **`Credit_Risk_Prediction.ipynb`** - Credit risk assessment
- **`Insurance_Policy_Renewal_Prediction.ipynb`** - Insurance policy renewal prediction
- **Usage**: `python Predictive_Modeling/Classification_Problems/classification_examples.py`

#### Regression Problems
- **`regression_examples.py`** - Advanced regression techniques and feature selection
- **`Predicting_South_Korean_Condominium_Prices.ipynb`** - Real estate price prediction
- **Usage**: `python Predictive_Modeling/Regression_Problems/regression_examples.py`

#### Recommendation Systems
- **`recommendation_examples.py`** - Collaborative filtering, content-based, and hybrid approaches
- **`Segmentation_Recommendation_Online_Investment_Platforms.ipynb`** - Investment platform recommendations
- **Usage**: `python Predictive_Modeling/Recommendation_Systems/recommendation_examples.py`

#### Anomaly Detection
- **`anomaly_detection_examples.py`** - Isolation Forest, One-Class SVM, and autoencoders
- **Usage**: `python Predictive_Modeling/Anomaly_Detection/anomaly_detection_examples.py`

#### Multi-Label Classification
- **`multi_label_examples.py`** - Multi-label classification with label powerset and classifier chains
- **Usage**: `python Predictive_Modeling/Multi_Label_Classification/multi_label_examples.py`

### 🎯 Clustering

<a name="clustering"></a>
Unsupervised learning and data segmentation.

#### Clustering Examples
- **`clustering_examples.py`** - K-means, hierarchical clustering, DBSCAN, and customer segmentation
- **Usage**: `python Clustering/clustering_examples.py`

### 👁️ Computer Vision

<a name="computer-vision"></a>
Image processing and computer vision applications.

#### Image Classification
- **`image_classification_examples.py`** - CNN architectures, transfer learning, and image preprocessing
- **Usage**: `python Computer_Vision/Image_Classification/image_classification_examples.py`

#### Object Detection
- **`object_detection_examples.py`** - YOLO, Faster R-CNN, and custom object detection models
- **Usage**: `python Computer_Vision/Object_Detection/object_detection_examples.py`

#### Image Segmentation
- **`image_segmentation_examples.py`** - U-Net, Mask R-CNN, and semantic segmentation
- **Usage**: `python Computer_Vision/Image_Segmentation/image_segmentation_examples.py`

#### Face Recognition
- **`face_recognition_examples.py`** - Face detection, recognition, and verification systems
- **Usage**: `python Computer_Vision/Face_Recognition/face_recognition_examples.py`

#### Feature Extraction
- **`feature_extraction_examples.py`** - SIFT, SURF, ORB, and deep feature extraction
- **Usage**: `python Computer_Vision/Feature_Extraction/feature_extraction_examples.py`

### 🚀 MLOps

<a name="mlops"></a>
Machine Learning Operations and production deployment.

#### Model Pipeline
- **`ml_pipeline_examples.py`** - Scikit-learn pipelines, MLflow integration, and automated workflows
- **Usage**: `python MLOps/Model_Pipeline/ml_pipeline_examples.py`

#### Model Deployment
- **`model_deployment_examples.py`** - Flask/FastAPI APIs, Docker containerization, and cloud deployment
- **Usage**: `python MLOps/Model_Deployment/model_deployment_examples.py`

## 🛠️ Technologies & Libraries

<a name="technologies-libraries"></a>

### Core ML Libraries
- **Scikit-learn**: Traditional ML algorithms
- **XGBoost/LightGBM/CatBoost**: Gradient boosting frameworks
- **NumPy/Pandas**: Numerical computing and data manipulation
- **Matplotlib/Seaborn**: Data visualization

### Deep Learning
- **TensorFlow/Keras**: Neural network implementation
- **PyTorch**: Alternative deep learning framework
- **Transformers**: Pre-trained language models

### NLP
- **NLTK**: Natural language processing
- **spaCy**: Industrial-strength NLP
- **Gensim**: Topic modeling and word embeddings

### Computer Vision
- **OpenCV**: Computer vision library
- **Pillow**: Image processing
- **Albumentations**: Image augmentation

### Optimization
- **SciPy**: Scientific computing and optimization
- **CVXPY**: Convex optimization
- **PuLP**: Linear programming

### MLOps
- **MLflow**: Experiment tracking and model management
- **Flask/FastAPI**: Web API frameworks
- **Docker**: Containerization

## 📚 Learning Path

<a name="learning-path"></a>

### Beginner Level
1. Start with **Basic Machine Learning** folder
2. Learn data preprocessing and EDA
3. Practice with regression and classification
4. Explore ensemble methods

### Intermediate Level
1. Dive into **Deep Learning** fundamentals
2. Work with **NLP** text processing
3. Implement **Computer Vision** applications
4. Study **Optimization** techniques

### Advanced Level
1. Build **Predictive Modeling** solutions
2. Implement **Clustering** algorithms
3. Deploy models with **MLOps**
4. Create end-to-end ML pipelines

## 🎯 Use Cases

<a name="use-cases"></a>

### Business Applications
- Customer segmentation and targeting
- Fraud detection and risk assessment
- Demand forecasting and inventory management
- Recommendation systems for e-commerce

### Research Applications
- Scientific data analysis
- Image and text processing
- Time series forecasting
- Anomaly detection in complex systems

### Production Systems
- Real-time prediction APIs
- Automated ML pipelines
- Model monitoring and maintenance
- Scalable ML infrastructure

## 🔧 Customization

<a name="customization"></a>

### Adding New Algorithms
```python
# Create a new file in the appropriate folder
# Follow the existing code structure
# Include comprehensive documentation
# Add example usage and parameter explanations
```

### Extending Existing Examples
```python
# Import the base functionality
from existing_module import BaseClass

# Extend with your custom implementation
class CustomImplementation(BaseClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add your custom logic
```

## 📝 Contributing

<a name="contributing"></a>

1. Fork the repository
2. Create a feature branch
3. Add your implementation with proper documentation
4. Include example usage and test cases
5. Submit a pull request

## 📄 License

<a name="license"></a>

This project is created for educational purposes. Feel free to use and modify for learning and research.

## 👤 Author

<a name="author"></a>

**Leanna Jeon**

## 🙏 Acknowledgments

- Open source ML community
- Academic institutions and research papers
- Industry best practices and case studies

---

**Note**: This repository contains practical implementations of machine learning algorithms and techniques. Each script is designed to be runnable and includes comprehensive documentation, making it suitable for both learning and production use. 