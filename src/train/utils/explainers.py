"""
Explainability Module

Provides model explanation tools:
- SHAP values computation
- Feature importance (for tree and neural models)
- Visualization generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any, Optional, List, Tuple
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logger = logging.getLogger(__name__)


class ExplainerWrapper:
    """
    Wrapper for model explainability across different model types.
    """
    
    def __init__(
        self,
        model: Any,
        model_type: str,
        feature_names: List[str],
        device: Optional[Any] = None
    ):
        """
        Initialize explainer.
        
        Args:
            model: Model instance
            model_type: Type of model
            feature_names: List of feature names
            device: Device for neural models
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.device = device
        self.shap_values = None
        self.expected_value = None
    
    def compute_shap_values(
        self,
        X_background: np.ndarray,
        X_explain: np.ndarray,
        sample_size: int = 2000,
        background_size: int = 512
    ) -> Tuple[np.ndarray, float]:
        """
        Compute SHAP values.
        
        Args:
            X_background: Background data for SHAP
            X_explain: Data to explain
            sample_size: Number of samples to explain
            background_size: Background data size
        
        Returns:
            tuple: (shap_values, expected_value)
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return None, None
        
        logger.info(f"Computing SHAP values for {self.model_type}")
        
        # Sample data if needed
        if len(X_background) > background_size:
            indices = np.random.choice(len(X_background), background_size, replace=False)
            X_background = X_background[indices]
        
        if len(X_explain) > sample_size:
            indices = np.random.choice(len(X_explain), sample_size, replace=False)
            X_explain = X_explain[indices]
        
        # Choose appropriate explainer
        if self.model_type in ['xgboost', 'random_forest']:
            # TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_explain)
            expected_value = explainer.expected_value
            
            # Handle multi-output (get positive class)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
        elif self.model_type in ['mlp', 'nn', 'wide_deep', 'tab_transformer']:
            # KernelExplainer for neural networks
            import torch
            
            def model_predict(x):
                self.model.eval()
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x).to(self.device)
                    outputs = torch.sigmoid(self.model(x_tensor)).cpu().numpy()
                return outputs
            
            explainer = shap.KernelExplainer(model_predict, X_background)
            shap_values = explainer.shap_values(X_explain, nsamples=100)
            expected_value = explainer.expected_value
        
        else:
            raise ValueError(f"SHAP not supported for model type: {self.model_type}")
        
        self.shap_values = shap_values
        self.expected_value = expected_value
        
        logger.info(f"SHAP values computed: shape={shap_values.shape}")
        return shap_values, expected_value
    
    def plot_shap_summary(
        self,
        X: np.ndarray,
        save_path: Optional[Path] = None,
        max_display: int = 20
    ):
        """
        Plot SHAP summary plot.
        
        Args:
            X: Feature data
            save_path: Path to save plot
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            logger.warning("SHAP values not computed yet")
            return
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved: {save_path}")
        plt.close()
    
    def plot_shap_bar(
        self,
        save_path: Optional[Path] = None,
        top_k: int = 20
    ):
        """
        Plot SHAP bar plot of feature importance.
        
        Args:
            save_path: Path to save plot
            top_k: Number of top features
        """
        if self.shap_values is None:
            logger.warning("SHAP values not computed yet")
            return
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=top_k,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP bar plot saved: {save_path}")
        plt.close()
    
    def get_feature_importance_shap(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance from SHAP values.
        
        Args:
            top_k: Number of top features to return
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.shap_values is None:
            logger.warning("SHAP values not computed yet")
            return None
        
        # Compute mean absolute SHAP values
        importance = np.abs(self.shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        if top_k:
            df = df.head(top_k)
        
        return df


def compute_feature_importance(
    model: Any,
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = 'auto'
) -> pd.DataFrame:
    """
    Compute feature importance.
    
    Args:
        model: Model instance
        model_type: Type of model
        X: Feature data
        y: Target data
        feature_names: List of feature names
        method: Importance method ('auto', 'gain', 'permutation')
    
    Returns:
        pd.DataFrame: Feature importance
    """
    logger.info(f"Computing feature importance: method={method}")
    
    if method == 'auto':
        if model_type in ['xgboost', 'random_forest']:
            method = 'gain'
        else:
            method = 'permutation'
    
    if method == 'gain' and model_type == 'xgboost':
        # XGBoost feature importance
        importance = model.feature_importances_
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
    
    elif method == 'gain' and model_type == 'random_forest':
        # Random Forest feature importance
        importance = model.feature_importances_
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
    
    elif method == 'permutation':
        # Permutation importance (model-agnostic)
        from sklearn.inspection import permutation_importance
        
        # For neural networks, need to wrap predict_proba
        if model_type in ['mlp', 'nn', 'wide_deep', 'tab_transformer']:
            import torch
            def predict_fn(X):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    outputs = torch.sigmoid(model(X_tensor)).cpu().numpy().flatten()
                return outputs
            
            # Use sklearn wrapper
            from sklearn.base import BaseEstimator
            class ModelWrapper(BaseEstimator):
                def __init__(self, predict_fn):
                    self.predict_fn = predict_fn
                def predict(self, X):
                    return self.predict_fn(X)
            
            wrapped_model = ModelWrapper(predict_fn)
            result = permutation_importance(
                wrapped_model, X, y,
                n_repeats=5,
                random_state=42,
                scoring='roc_auc'
            )
        else:
            result = permutation_importance(
                model, X, y,
                n_repeats=5,
                random_state=42,
                scoring='roc_auc'
            )
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': result.importances_mean,
            'importance_std': result.importances_std
        })
    
    else:
        raise ValueError(f"Unknown importance method: {method}")
    
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    logger.info(f"Feature importance computed: {len(df)} features")
    
    return df


def generate_shap_plots(
    explainer: ExplainerWrapper,
    X: np.ndarray,
    output_dir: Path,
    top_k: int = 20
):
    """
    Generate all SHAP plots.
    
    Args:
        explainer: ExplainerWrapper with computed SHAP values
        X: Feature data
        output_dir: Directory to save plots
        top_k: Number of top features
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary plot
    explainer.plot_shap_summary(X, output_dir / 'shap_summary.png', max_display=top_k)
    
    # Bar plot
    explainer.plot_shap_bar(output_dir / 'shap_bar.png', top_k=top_k)
    
    logger.info(f"SHAP plots saved to {output_dir}")

