import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.protocol_encoder = LabelEncoder()
        self.feature_columns = [
            'src_port', 'dst_port', 'protocol', 'flow_duration', 'pkt_count',
            'pkt_size_avg', 'bytes_sent', 'bytes_received', 'conn_count_last_10s',
            'same_dst_count', 'srv_serror_rate', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'entropy', 'honeypot_flag'
        ]
        
        # Protocol mapping
        self.protocol_mapping = {
            'TCP': 0,
            'UDP': 1,
            'ICMP': 2,
            'HTTP': 3,
            'HTTPS': 4,
            'DNS': 5,
            'OTHER': 6
        }
        
    def load_data(self, file_path):
        """Load and preprocess the dataset"""
        df = pd.read_csv(file_path)
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for model training"""
        # Create a copy of the dataframe
        df_processed = df.copy()
        
        # Ensure we only use the features we need
        df_processed = df_processed[self.feature_columns + ['is_malicious']]
        
        # Handle protocol column - map string values to numbers
        if 'protocol' in df_processed.columns:
            df_processed['protocol'] = df_processed['protocol'].map(
                lambda x: self.protocol_mapping.get(str(x).upper(), 6)
            )
        
        # Separate features and target
        X = df_processed[self.feature_columns]
        y = df_processed['is_malicious']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_scaler(self, path='models/scaler.joblib'):
        """Save the scaler for later use"""
        joblib.dump(self.scaler, path)
        # Save protocol mapping
        joblib.dump(self.protocol_mapping, path.replace('scaler.joblib', 'protocol_mapping.joblib'))
    
    def load_scaler(self, path='models/scaler.joblib'):
        """Load the saved scaler"""
        self.scaler = joblib.load(path)
        # Load protocol mapping
        try:
            self.protocol_mapping = joblib.load(path.replace('scaler.joblib', 'protocol_mapping.joblib'))
        except:
            pass  # Use default mapping if file doesn't exist
    
    def transform_single_input(self, input_data):
        """Transform a single input for prediction"""
        # Ensure input data has the correct features in the correct order
        if isinstance(input_data, np.ndarray):
            if input_data.shape[1] != len(self.feature_columns):
                raise ValueError(f"Expected {len(self.feature_columns)} features, but got {input_data.shape[1]}")
        
        # Convert input_data to numeric values
        input_data = input_data.astype(float)
        return self.scaler.transform(input_data) 