"""
Data processing module for predictive maintenance pipeline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import yaml
import logging
from typing import Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handle all data processing operations"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        logger.info("Creating engineered features")
        df = df.copy()
        
        # Temperature difference
        df['Temperature Difference [K]'] = (
            df['Process temperature [K]'] - df['Air temperature [K]']
        )
        
        # Power calculation (in Watts)
        df['Power [W]'] = (
            (df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi) / 60
        )
        
        # Tool wear rate (wear per unit produced)
        # Use UDI if available, otherwise use a default value
        if 'UDI' in df.columns:
            df['Tool Wear Rate'] = df['Tool wear [min]'] / (df['UDI'] + 1)
        else:
            df['Tool Wear Rate'] = df['Tool wear [min]'] / 100  # Default normalization
        
        # Tool wear categories
        df['Tool Wear Category'] = pd.cut(
            df['Tool wear [min]'],
            bins=[0, 50, 150, 300],
            labels=['Low', 'Medium', 'High']
        )
        
        # Torque-speed ratio
        df['Torque Speed Ratio'] = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1)
        
        # Temperature stress indicator
        df['Temp Stress'] = (
            (df['Process temperature [K]'] - 308) / 5 +  # Normalized process temp
            (df['Air temperature [K]'] - 298) / 5  # Normalized air temp
        )
        
        logger.info(f"Features created. New shape: {df.shape}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        df = df.copy()
        
        # One-hot encode product type
        df = pd.get_dummies(df, columns=['Type'], prefix='Type')
        
        # Encode tool wear category if it exists
        if 'Tool Wear Category' in df.columns:
            df['Tool_Wear_Low'] = (df['Tool Wear Category'] == 'Low').astype(int)
            df['Tool_Wear_Medium'] = (df['Tool Wear Category'] == 'Medium').astype(int)
            df['Tool_Wear_High'] = (df['Tool Wear Category'] == 'High').astype(int)
            df = df.drop('Tool Wear Category', axis=1)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling"""
        # Get feature columns
        feature_cols = (
            self.config['features']['numerical'] + 
            self.config['features']['engineered']
        )
        
        # Add encoded categorical columns
        type_cols = [col for col in df.columns if col.startswith('Type_')]
        tool_wear_cols = [col for col in df.columns if col.startswith('Tool_Wear_')]
        
        all_features = feature_cols + type_cols + tool_wear_cols
        
        # Filter to existing columns
        all_features = [col for col in all_features if col in df.columns]
        
        X = df[all_features]
        y = df[self.config['target']]
        
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
        return X, y
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=y if stratify else None
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Train failure rate: {y_train.mean()*100:.2f}%")
        logger.info(f"Test failure rate: {y_test.mean()*100:.2f}%")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numerical features"""
        # Get numerical columns
        num_cols = [col for col in self.config['features']['numerical'] + 
                   self.config['features']['engineered'] 
                   if col in X_train.columns]
        
        # Fit scaler on training data
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[num_cols] = self.scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = self.scaler.transform(X_test[num_cols])
        
        logger.info("Features scaled")
        return X_train_scaled, X_test_scaled
    
    def handle_imbalance(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        method: str = 'smote'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance"""
        
        if method == 'smote':
            logger.info("Applying SMOTE for class balancing")
            smote = SMOTE(random_state=self.config['model']['random_state'])
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            logger.info(f"After SMOTE - Train set: {len(X_train_balanced)} samples")
            logger.info(f"After SMOTE - Failure rate: {y_train_balanced.mean()*100:.2f}%")
            
            return pd.DataFrame(X_train_balanced, columns=X_train.columns), y_train_balanced
        
        return X_train, y_train
    
    def get_feature_names(self, X: pd.DataFrame) -> list:
        """Get list of feature names"""
        return X.columns.tolist()
    
    def process_pipeline(
        self, 
        filepath: str,
        handle_imbalance: bool = True
    ) -> Dict[str, Any]:
        """Complete processing pipeline"""
        
        # Load and process data
        df = self.load_data(filepath)
        df = self.create_features(df)
        df = self.encode_categorical(df)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Handle imbalance if requested
        if handle_imbalance:
            X_train_final, y_train_final = self.handle_imbalance(
                X_train_scaled, y_train
            )
        else:
            X_train_final, y_train_final = X_train_scaled, y_train
        
        return {
            'X_train': X_train_final,
            'X_test': X_test_scaled,
            'y_train': y_train_final,
            'y_test': y_test,
            'feature_names': self.get_feature_names(X),
            'scaler': self.scaler,
            'original_df': df
        }


if __name__ == "__main__":
    # Test the processor
    processor = DataProcessor()
    data = processor.process_pipeline("data/raw/predictive_maintenance.csv")
    print(f"Processing complete. Train shape: {data['X_train'].shape}")