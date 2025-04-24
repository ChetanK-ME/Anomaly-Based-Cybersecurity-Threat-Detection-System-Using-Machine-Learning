# Anomaly-Based Cybersecurity Threat Detection System

This project implements a machine learning-based system for detecting cybersecurity threats and anomalies in network traffic. The system uses multiple anomaly detection algorithms and automatically selects the best performing one based on the data.

## Features

- Multiple anomaly detection algorithms:
  - Isolation Forest: An ensemble-based approach that isolates anomalies by randomly selecting features and split values
  - Autoencoder: A deep learning approach that learns to compress and reconstruct normal data, identifying anomalies as data points with high reconstruction error
  - One-Class SVM: A kernel-based approach that learns a decision boundary around normal data points
- Automatic model selection based on performance metrics
- Interactive web interface using Streamlit
- Real-time anomaly detection
- Comprehensive performance analysis and visualization

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── data_preprocessing.py  # Data preprocessing module
├── model_training.py      # Model training and evaluation module
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Setup and Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Model Training**:
   - Navigate to the "Model Training" page
   - Click "Train Model" to train the anomaly detection models
   - View training metrics and model performance
   - Compare performance across different algorithms

2. **Anomaly Detection**:
   - Navigate to the "Anomaly Detection" page
   - Input the network traffic features
   - Click "Detect Anomaly" to get the prediction
   - View probability scores for normal and anomalous classes
   - For autoencoder-based detection, view the reconstruction error

3. **Performance Analysis**:
   - View detailed performance metrics for all models
   - Compare model performance across different metrics
   - Analyze feature importance (for Isolation Forest)
   - View ROC curves and AUC scores
   - Examine confusion matrices
   - View algorithm-specific details and parameters

## Dataset

The system uses a synthetic dataset containing 10,000 rows of network traffic data. The dataset includes various features related to network traffic patterns and is labeled with normal and anomalous traffic.

## Model Selection

The system automatically selects the best performing model based on the following metrics:
- Accuracy: Overall correctness of predictions
- Precision: Ability to avoid false positives
- Recall: Ability to find all anomalies
- F1 Score: Harmonic mean of precision and recall
- AUC-ROC: Area under the ROC curve, measuring model's ability to distinguish between classes
- Log Loss: Logarithmic loss, measuring the performance of a classification model

## Evaluation Metrics

The system evaluates models using a comprehensive set of metrics:

1. **Accuracy**: The proportion of correct predictions (both normal and anomalous) out of all predictions.
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Precision**: The proportion of correctly identified anomalies out of all predicted anomalies.
   ```
   Precision = TP / (TP + FP)
   ```

3. **Recall**: The proportion of correctly identified anomalies out of all actual anomalies.
   ```
   Recall = TP / (TP + FN)
   ```

4. **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.
   ```
   F1 = 2 * (Precision * Recall) / (Precision + Recall)
   ```

5. **AUC-ROC**: The area under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between normal and anomalous traffic across different classification thresholds.

6. **Log Loss**: A measure of the model's confidence in its predictions, penalizing incorrect predictions more heavily when the model is confident.
   ```
   Log Loss = -1/N * Σ(y_i * log(p_i) + (1-y_i) * log(1-p_i))
   ```

7. **Confusion Matrix**: A table showing the distribution of predictions across actual and predicted classes.

## Algorithm Details

### Isolation Forest
- An ensemble-based approach that isolates anomalies by randomly selecting features and split values
- Parameters:
  - Contamination: 0.1 (expected proportion of anomalies)
  - Random State: 42 (for reproducibility)
  - Number of Trees: 100 (default)

### Autoencoder
- A deep learning approach that learns to compress and reconstruct normal data
- Architecture:
  - Input Layer: 10 neurons (matching input features)
  - Encoder: 32 neurons with ReLU activation
  - Decoder: 10 neurons with Sigmoid activation
  - Dropout: 0.2 for regularization
- Anomaly detection based on reconstruction error threshold

### One-Class SVM
- A kernel-based approach that learns a decision boundary around normal data points
- Parameters:
  - Kernel: RBF (Radial Basis Function)
  - Nu: 0.1 (upper bound on the fraction of outliers)
  - Gamma: scale (default)

## Limitations and Improvements

This implementation addresses several limitations found in previous research:

1. **Multiple Algorithm Support**: Unlike previous implementations that focused on single algorithms, this system implements multiple algorithms and automatically selects the best one.

2. **Deep Learning Integration**: The addition of the Autoencoder provides a deep learning approach that can capture complex patterns in the data.

3. **Comprehensive Evaluation**: The system provides a wide range of evaluation metrics for better understanding of model performance.

4. **Real-time Processing**: The system is designed for real-time anomaly detection, making it suitable for production environments.

5. **User-friendly Interface**: The Streamlit interface makes it easy for users to interact with the system without technical knowledge.

## Future Improvements

1. Add more anomaly detection algorithms (e.g., DBSCAN, LOF)
2. Implement feature selection techniques
3. Add support for real-time data streaming
4. Enhance visualization capabilities
5. Add model explanation features
6. Implement ensemble methods combining multiple algorithms
7. Add support for different types of network traffic data

## Contributing

Feel free to submit issues and enhancement requests! 