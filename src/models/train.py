# File: src/models/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

def prepare_data(df, target_column='AttritionFlag'):
    """
    Prepare data for modeling by separating features and target
    """
    X = df.drop(columns=[target_column, 'Attrition', 'EmployeeID', 'Name'], errors='ignore')
    y = df[target_column]
    
    return X, y

def get_feature_types(X):
    """
    Identify numerical and categorical features
    """
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return numerical_features, categorical_features

def create_preprocessor(numerical_features, categorical_features):
    """
    Create preprocessing pipeline for numerical and categorical features
    """
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_models(X, y, test_size=0.2, random_state=42):
    """
    Train multiple models and return results
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Get feature types
    numerical_features, categorical_features = get_feature_types(X)
    
    # Create preprocessor
    preprocessor = create_preprocessor(numerical_features, categorical_features)
    
    # Define models
    models = {
        'RandomForest': RandomForestClassifier(random_state=random_state, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(random_state=random_state),
        'LogisticRegression': LogisticRegression(random_state=random_state, class_weight='balanced', max_iter=1000),
    }
    
    results = {}
    
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'model': pipeline,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"{name}: Accuracy = {accuracy:.3f}, F1 = {f1:.3f}, ROC-AUC = {roc_auc:.3f}")
    
    return results, X_test, y_test

def optimize_random_forest(X, y):
    """
    Optimize Random Forest using GridSearchCV
    """
    numerical_features, categorical_features = get_feature_types(X)
    preprocessor = create_preprocessor(numerical_features, categorical_features)
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    # Define parameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X, y)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best F1 score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def save_model(model, path):
    """
    Save trained model to file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved!!")

def plot_feature_importance(model, X, save_path=None):
    """
    Plot feature importance for the best model
    """
    # Get feature names after preprocessing
    preprocessor = model.named_steps['preprocessor']
    feature_names = []
    
    # Numerical features
    feature_names.extend(X.select_dtypes(include=['int64', 'float64']).columns.tolist())
    
    # Categorical features (after one-hot encoding)
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for feature in categorical_features:
        categories = X[feature].unique()
        for category in categories:
            feature_names.append(f"{feature}_{category}")
    
    # Get feature importance
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        })
        
        # Sort and plot top 20 features
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance

def evaluate_best_model(best_model, X_test, y_test):
    """
    Evaluate the best model on test data
    """
    # Predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Best Model Evaluation:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc
    }