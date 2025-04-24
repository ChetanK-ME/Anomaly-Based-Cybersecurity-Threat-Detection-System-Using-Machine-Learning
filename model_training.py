import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class AutoencoderAnomalyDetector:
    def __init__(self, input_dim, encoding_dim=32, threshold=0.1):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.threshold = threshold
        self.model = self._build_model()
        
    def _build_model(self):
        # Encoder
        input_layer = Input(shape=(self.input_dim,))
        encoder = Dense(self.encoding_dim, activation='relu')(input_layer)
        encoder = Dropout(0.2)(encoder)
        
        # Decoder
        decoder = Dense(self.input_dim, activation='sigmoid')(encoder)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoder)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return autoencoder
    
    def fit(self, X, epochs=50, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate reconstruction error threshold
        reconstructed = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        self.threshold = np.percentile(mse, 90)  # 90th percentile as threshold
        
        return self
    
    def predict(self, X):
        reconstructed = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        return np.where(mse > self.threshold, -1, 1)  # -1 for anomaly, 1 for normal
    
    def predict_proba(self, X):
        """Return probability scores for anomaly detection"""
        reconstructed = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        # Normalize MSE to [0,1] range for probability
        anomaly_prob = 1 - (mse - np.min(mse)) / (np.max(mse) - np.min(mse) + 1e-10)
        normal_prob = 1 - anomaly_prob
        return np.vstack([normal_prob, anomaly_prob]).T  # Return [normal_prob, anomaly_prob]
    
    def save_model(self, path):
        self.model.save(path)
    
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

class AnomalyDetector:
    def __init__(self):
        self.models = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'one_class_svm': OneClassSVM(kernel='rbf', nu=0.1),
            'autoencoder': None  # Will be initialized when training
        }
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        
    def train_models(self, X_train, y_train):
        """Train all models and select the best one"""
        best_f1 = -1
        
        # Train Isolation Forest
        self.models['isolation_forest'].fit(X_train)
        y_pred_if = self.models['isolation_forest'].predict(X_train)
        y_pred_if = np.where(y_pred_if == 1, 0, 1)
        
        # Get probability scores for Isolation Forest
        y_score_if = -self.models['isolation_forest'].score_samples(X_train)
        y_score_if = (y_score_if - np.min(y_score_if)) / (np.max(y_score_if) - np.min(y_score_if))
        y_prob_if = np.vstack([1-y_score_if, y_score_if]).T
        
        metrics_if = {
            'accuracy': accuracy_score(y_train, y_pred_if),
            'precision': precision_score(y_train, y_pred_if),
            'recall': recall_score(y_train, y_pred_if),
            'f1': f1_score(y_train, y_pred_if),
            'auc_roc': roc_auc_score(y_train, y_score_if),
            'log_loss': log_loss(y_train, y_prob_if)
        }
        self.metrics['isolation_forest'] = metrics_if
        
        # Train One-Class SVM
        self.models['one_class_svm'].fit(X_train)
        y_pred_svm = self.models['one_class_svm'].predict(X_train)
        y_pred_svm = np.where(y_pred_svm == 1, 0, 1)
        
        # Get probability scores for One-Class SVM using decision function
        y_score_svm = -self.models['one_class_svm'].decision_function(X_train)
        y_score_svm = (y_score_svm - np.min(y_score_svm)) / (np.max(y_score_svm) - np.min(y_score_svm))
        y_prob_svm = np.vstack([1-y_score_svm, y_score_svm]).T
        
        metrics_svm = {
            'accuracy': accuracy_score(y_train, y_pred_svm),
            'precision': precision_score(y_train, y_pred_svm),
            'recall': recall_score(y_train, y_pred_svm),
            'f1': f1_score(y_train, y_pred_svm),
            'auc_roc': roc_auc_score(y_train, y_score_svm),
            'log_loss': log_loss(y_train, y_prob_svm)
        }
        self.metrics['one_class_svm'] = metrics_svm
        
        # Train Autoencoder
        self.models['autoencoder'] = AutoencoderAnomalyDetector(input_dim=X_train.shape[1])
        self.models['autoencoder'].fit(X_train)
        y_pred_ae = self.models['autoencoder'].predict(X_train)
        y_pred_ae = np.where(y_pred_ae == 1, 0, 1)
        
        # Get probability scores for Autoencoder
        y_prob_ae = self.models['autoencoder'].predict_proba(X_train)
        y_score_ae = y_prob_ae[:, 1]  # Probability of anomaly
        
        metrics_ae = {
            'accuracy': accuracy_score(y_train, y_pred_ae),
            'precision': precision_score(y_train, y_pred_ae),
            'recall': recall_score(y_train, y_pred_ae),
            'f1': f1_score(y_train, y_pred_ae),
            'auc_roc': roc_auc_score(y_train, y_score_ae),
            'log_loss': log_loss(y_train, y_prob_ae)
        }
        self.metrics['autoencoder'] = metrics_ae
        
        # Select best model
        for name, metrics in self.metrics.items():
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                self.best_model = self.models[name]
                self.best_model_name = name
        
        return self.best_model_name, self.metrics
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the best model on test data"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        y_pred = self.best_model.predict(X_test)
        y_pred = np.where(y_pred == 1, 0, 1)
        
        # Get probability scores based on model type
        if self.best_model_name == 'isolation_forest':
            y_score = -self.best_model.score_samples(X_test)
            y_score = (y_score - np.min(y_score)) / (np.max(y_score) - np.min(y_score))
            y_prob = np.vstack([1-y_score, y_score]).T
        elif self.best_model_name == 'one_class_svm':
            y_score = -self.best_model.decision_function(X_test)
            y_score = (y_score - np.min(y_score)) / (np.max(y_score) - np.min(y_score))
            y_prob = np.vstack([1-y_score, y_score]).T
        else:  # autoencoder
            y_prob = self.best_model.predict_proba(X_test)
            y_score = y_prob[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_score),
            'log_loss': log_loss(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return test_metrics
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """Get probability scores using the best model"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        if self.best_model_name == 'isolation_forest':
            y_score = -self.best_model.score_samples(X)
            y_score = (y_score - np.min(y_score)) / (np.max(y_score) - np.min(y_score))
            return np.vstack([1-y_score, y_score]).T
        elif self.best_model_name == 'one_class_svm':
            y_score = -self.best_model.decision_function(X)
            y_score = (y_score - np.min(y_score)) / (np.max(y_score) - np.min(y_score))
            return np.vstack([1-y_score, y_score]).T
        else:  # autoencoder
            return self.best_model.predict_proba(X)
    
    def save_model(self, path='models/best_model'):
        """Save the best model"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.best_model_name == 'autoencoder':
            self.best_model.save_model(path)
        else:
            joblib.dump(self.best_model, path + '.joblib')
        
        # Save model name
        with open(path + '_name.txt', 'w') as f:
            f.write(self.best_model_name)
    
    def load_model(self, path='models/best_model'):
        """Load the saved model"""
        # Load model name
        with open(path + '_name.txt', 'r') as f:
            self.best_model_name = f.read().strip()
        
        if self.best_model_name == 'autoencoder':
            self.models['autoencoder'] = AutoencoderAnomalyDetector(input_dim=0)  # Dummy input_dim
            self.models['autoencoder'].load_model(path)
            self.best_model = self.models['autoencoder']
        else:
            self.best_model = joblib.load(path + '.joblib') 