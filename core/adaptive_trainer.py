from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Adaptive Trainer - Machine Learning Model Training and Adaptation for Schwabot
==============================================================================

This module implements the adaptive trainer for Schwabot, providing
comprehensive machine learning model training, adaptation, and optimization
for the trading system.

Core Functionality:
- Adaptive model training
- Online learning and adaptation
- Performance monitoring and optimization
- Model versioning and management
- Hyperparameter optimization
"""

import logging
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from core.unified_math_system import unified_math
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class TrainingMode(Enum):
    BATCH = "batch"
    ONLINE = "online"
    INCREMENTAL = "incremental"
    TRANSFER = "transfer"
    META = "meta"

class ModelStatus(Enum):
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"

@dataclass
class TrainingConfig:
    config_id: str
    model_type: str
    training_mode: TrainingMode
    hyperparameters: Dict[str, Any]
    data_config: Dict[str, Any]
    validation_config: Dict[str, Any]
    optimization_config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingResult:
    result_id: str
    model_id: str
    training_config: TrainingConfig
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    metrics: Dict[str, float]
    model_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelVersion:
    version_id: str
    model_id: str
    version_number: str
    model_path: str
    training_result: TrainingResult
    performance_metrics: Dict[str, float]
    deployment_status: ModelStatus
    created_at: datetime
    deployed_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdaptiveTrainer:
    def __init__(self, config_path: str = "./config/adaptive_trainer_config.json"):
        self.config_path = config_path
        self.training_configs: Dict[str, TrainingConfig] = {}
        self.training_results: Dict[str, TrainingResult] = {}
        self.model_versions: Dict[str, ModelVersion] = {}
        self.active_models: Dict[str, Any] = {}
        self.training_queue: deque = deque(maxlen=100)
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self._load_configuration()
        self._initialize_trainer()
        self._start_training_monitor()
        logger.info("AdaptiveTrainer initialized")

    def _load_configuration(self) -> None:
        """Load adaptive trainer configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                logger.info(f"Loaded adaptive trainer configuration")
            else:
                self._create_default_configuration()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_configuration()

    def _create_default_configuration(self) -> None:
        """Create default adaptive trainer configuration."""
        config = {
            "training_modes": {
                "batch": {
                    "batch_size": 1000,
                    "epochs": 100,
                    "learning_rate": 0.001
                },
                "online": {
                    "learning_rate": 0.01,
                    "update_frequency": 100,
                    "memory_size": 10000
                },
                "incremental": {
                    "chunk_size": 500,
                    "retrain_frequency": 24,
                    "performance_threshold": 0.8
                }
            },
            "model_types": {
                "profit_predictor": {
                    "architecture": "neural_network",
                    "layers": [64, 32, 16, 1],
                    "activation": "relu"
                },
                "risk_assessor": {
                    "architecture": "gradient_boosting",
                    "n_estimators": 100,
                    "max_depth": 6
                },
                "opportunity_detector": {
                    "architecture": "random_forest",
                    "n_estimators": 200,
                    "max_features": "sqrt"
                }
            },
            "optimization": {
                "hyperparameter_tuning": True,
                "cross_validation_folds": 5,
                "optimization_algorithm": "bayesian"
            }
        }
        
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def _initialize_trainer(self) -> None:
        """Initialize the adaptive trainer."""
        # Initialize training environments
        self._initialize_training_environments()
        
        # Initialize model registry
        self._initialize_model_registry()
        
        logger.info("Adaptive trainer initialized successfully")

    def _initialize_training_environments(self) -> None:
        """Initialize training environments for different modes."""
        try:
            # Create training environments for different modes
            self.training_environments = {
                TrainingMode.BATCH: {
                    "status": "ready",
                    "resources": {"cpu": 4, "memory": "8GB", "gpu": 1},
                    "queue_size": 0
                },
                TrainingMode.ONLINE: {
                    "status": "ready",
                    "resources": {"cpu": 2, "memory": "4GB", "gpu": 0},
                    "queue_size": 0
                },
                TrainingMode.INCREMENTAL: {
                    "status": "ready",
                    "resources": {"cpu": 2, "memory": "4GB", "gpu": 0},
                    "queue_size": 0
                }
            }
            
            logger.info(f"Initialized {len(self.training_environments)} training environments")
            
        except Exception as e:
            logger.error(f"Error initializing training environments: {e}")

    def _initialize_model_registry(self) -> None:
        """Initialize model registry."""
        try:
            self.model_registry = {
                "models": {},
                "versions": {},
                "deployments": {},
                "performance_tracking": {}
            }
            
            logger.info("Model registry initialized")
            
        except Exception as e:
            logger.error(f"Error initializing model registry: {e}")

    def _start_training_monitor(self) -> None:
        """Start the training monitoring system."""
        # This would start background monitoring tasks
        logger.info("Training monitor started")

    def create_training_config(self, model_type: str, training_mode: TrainingMode,
                             hyperparameters: Optional[Dict[str, Any]] = None,
                             data_config: Optional[Dict[str, Any]] = None) -> str:
        """Create a new training configuration."""
        try:
            config_id = f"config_{model_type}_{training_mode.value}_{int(time.time())}"
            
            # Get default configuration for model type and mode
            default_hyperparams = self._get_default_hyperparameters(model_type, training_mode)
            default_data_config = self._get_default_data_config(model_type)
            
            # Merge with provided parameters
            final_hyperparams = {**default_hyperparams, **(hyperparameters or {})}
            final_data_config = {**default_data_config, **(data_config or {})}
            
            # Create training configuration
            training_config = TrainingConfig(
                config_id=config_id,
                model_type=model_type,
                training_mode=training_mode,
                hyperparameters=final_hyperparams,
                data_config=final_data_config,
                validation_config=self._get_validation_config(),
                optimization_config=self._get_optimization_config(),
                metadata={"created_at": datetime.now().isoformat()}
            )
            
            # Store configuration
            self.training_configs[config_id] = training_config
            
            logger.info(f"Created training configuration: {config_id}")
            return config_id
            
        except Exception as e:
            logger.error(f"Error creating training configuration: {e}")
            return ""

    def _get_default_hyperparameters(self, model_type: str, training_mode: TrainingMode) -> Dict[str, Any]:
        """Get default hyperparameters for model type and training mode."""
        try:
            # Load configuration
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            mode_config = config.get("training_modes", {}).get(training_mode.value, {})
            model_config = config.get("model_types", {}).get(model_type, {})
            
            # Combine mode and model specific parameters
            hyperparams = {**mode_config, **model_config}
            
            return hyperparams
            
        except Exception as e:
            logger.error(f"Error getting default hyperparameters: {e}")
            return {}

    def _get_default_data_config(self, model_type: str) -> Dict[str, Any]:
        """Get default data configuration for model type."""
        return {
            "data_source": "market_data",
            "features": ["price", "volume", "volatility", "trend"],
            "target": "profit",
            "train_split": 0.8,
            "validation_split": 0.1,
            "test_split": 0.1,
            "window_size": 100,
            "normalization": "standard"
        }

    def _get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return {
            "validation_metric": "accuracy",
            "cross_validation_folds": 5,
            "early_stopping_patience": 10,
            "min_delta": 0.001
        }

    def _get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        return {
            "optimizer": "adam",
            "loss_function": "mse",
            "learning_rate_schedule": "exponential_decay",
            "regularization": "l2",
            "dropout_rate": 0.2
        }

    async def start_training(self, config_id: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Start training with a specific configuration."""
        try:
            if config_id not in self.training_configs:
                logger.error(f"Training configuration {config_id} not found")
                return ""
            
            training_config = self.training_configs[config_id]
            
            # Create training result
            result_id = f"result_{config_id}_{int(time.time())}"
            training_result = TrainingResult(
                result_id=result_id,
                model_id=f"model_{training_config.model_type}_{int(time.time())}",
                training_config=training_config,
                start_time=datetime.now(),
                end_time=None,
                success=False,
                metrics={},
                model_path=None,
                error_message=None,
                metadata={"training_mode": training_config.training_mode.value}
            )
            
            # Store training result
            self.training_results[result_id] = training_result
            
            # Add to training queue
            self.training_queue.append({
                "result_id": result_id,
                "config": training_config,
                "data": data
            })
            
            logger.info(f"Started training: {result_id}")
            return result_id
            
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return ""

    async def train_model(self, result_id: str, training_config: TrainingConfig,
                         data: Optional[Dict[str, Any]] = None) -> bool:
        """Execute model training."""
        try:
            training_result = self.training_results[result_id]
            training_result.metadata["training_started"] = datetime.now().isoformat()
            
            logger.info(f"Training model: {result_id}")
            
            # Simulate training process
            training_duration = self._estimate_training_duration(training_config)
            await asyncio.sleep(training_duration)
            
            # Generate training metrics
            metrics = self._generate_training_metrics(training_config, data)
            training_result.metrics = metrics
            
            # Check if training was successful
            success = metrics.get("accuracy", 0) > 0.7  # 70% accuracy threshold
            training_result.success = success
            
            if success:
                # Create model version
                model_path = f"models/{training_result.model_id}_{int(time.time())}.pkl"
                training_result.model_path = model_path
                
                # Create model version
                version = ModelVersion(
                    version_id=f"v_{training_result.model_id}_{int(time.time())}",
                    model_id=training_result.model_id,
                    version_number="1.0.0",
                    model_path=model_path,
                    training_result=training_result,
                    performance_metrics=metrics,
                    deployment_status=ModelStatus.READY,
                    created_at=datetime.now(),
                    deployed_at=None,
                    metadata={"training_config_id": training_config.config_id}
                )
                
                self.model_versions[version.version_id] = version
                
                logger.info(f"Training completed successfully: {result_id}")
            else:
                training_result.error_message = "Training failed to meet accuracy threshold"
                logger.warning(f"Training failed: {result_id}")
            
            training_result.end_time = datetime.now()
            
            return success
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            training_result = self.training_results[result_id]
            training_result.success = False
            training_result.error_message = str(e)
            training_result.end_time = datetime.now()
            return False

    def _estimate_training_duration(self, training_config: TrainingConfig) -> float:
        """Estimate training duration based on configuration."""
        try:
            base_duration = 1.0  # 1 second base
            
            # Adjust based on training mode
            mode_multipliers = {
                TrainingMode.BATCH: 2.0,
                TrainingMode.ONLINE: 0.5,
                TrainingMode.INCREMENTAL: 1.5
            }
            
            mode_multiplier = mode_multipliers.get(training_config.training_mode, 1.0)
            
            # Adjust based on model complexity
            model_complexity = len(training_config.hyperparameters.get("layers", [64, 32, 16, 1]))
            complexity_multiplier = model_complexity / 4.0
            
            duration = base_duration * mode_multiplier * complexity_multiplier
            
            return unified_math.max(0.1, unified_math.min(5.0, duration))  # Between 0.1 and 5 seconds
            
        except Exception as e:
            logger.error(f"Error estimating training duration: {e}")
            return 1.0

    def _generate_training_metrics(self, training_config: TrainingConfig,
                                 data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Generate training metrics."""
        try:
            # Simulate training metrics
            base_accuracy = 0.75
            
            # Adjust based on model type
            model_type_boost = {
                "profit_predictor": 0.05,
                "risk_assessor": 0.03,
                "opportunity_detector": 0.04
            }
            
            accuracy_boost = model_type_boost.get(training_config.model_type, 0.0)
            accuracy = base_accuracy + accuracy_boost + (np.random.random() - 0.5) * 0.1
            
            # Generate other metrics
            metrics = {
                "accuracy": unified_math.max(0.0, unified_math.min(1.0, accuracy)),
                "precision": accuracy * 0.95,
                "recall": accuracy * 0.92,
                "f1_score": accuracy * 0.93,
                "loss": 1.0 - accuracy,
                "training_time": self._estimate_training_duration(training_config)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error generating training metrics: {e}")
            return {"accuracy": 0.0, "loss": 1.0}

    async def deploy_model(self, version_id: str) -> bool:
        """Deploy a model version."""
        try:
            if version_id not in self.model_versions:
                logger.error(f"Model version {version_id} not found")
                return False
            
            model_version = self.model_versions[version_id]
            
            # Check if model is ready for deployment
            if model_version.deployment_status != ModelStatus.READY:
                logger.error(f"Model {version_id} is not ready for deployment")
                return False
            
            # Simulate deployment process
            await asyncio.sleep(0.5)
            
            # Update deployment status
            model_version.deployment_status = ModelStatus.DEPLOYED
            model_version.deployed_at = datetime.now()
            
            # Add to active models
            self.active_models[model_version.model_id] = model_version
            
            logger.info(f"Model deployed successfully: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            return False

    async def adapt_model(self, model_id: str, new_data: Dict[str, Any]) -> bool:
        """Adapt an existing model with new data."""
        try:
            if model_id not in self.active_models:
                logger.error(f"Active model {model_id} not found")
                return False
            
            model_version = self.active_models[model_id]
            
            # Create adaptation training config
            adaptation_config = TrainingConfig(
                config_id=f"adapt_{model_id}_{int(time.time())}",
                model_type=model_version.training_result.training_config.model_type,
                training_mode=TrainingMode.INCREMENTAL,
                hyperparameters=model_version.training_result.training_config.hyperparameters,
                data_config=model_version.training_result.training_config.data_config,
                validation_config=model_version.training_result.training_config.validation_config,
                optimization_config=model_version.training_result.training_config.optimization_config,
                metadata={"adaptation": True, "base_model": model_id}
            )
            
            # Start adaptation training
            result_id = await self.start_training(adaptation_config.config_id, new_data)
            if not result_id:
                return False
            
            # Execute adaptation
            success = await self.train_model(result_id, adaptation_config, new_data)
            
            if success:
                # Create new version
                new_version_id = f"v_{model_id}_adapt_{int(time.time())}"
                new_version = ModelVersion(
                    version_id=new_version_id,
                    model_id=model_id,
                    version_number="1.1.0",
                    model_path=f"models/{model_id}_adapt_{int(time.time())}.pkl",
                    training_result=self.training_results[result_id],
                    performance_metrics=self.training_results[result_id].metrics,
                    deployment_status=ModelStatus.READY,
                    created_at=datetime.now(),
                    deployed_at=None,
                    metadata={"adaptation": True, "base_version": model_version.version_id}
                )
                
                self.model_versions[new_version_id] = new_version
                
                logger.info(f"Model adaptation completed: {new_version_id}")
                return True
            else:
                logger.error(f"Model adaptation failed: {result_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adapting model: {e}")
            return False

    def get_training_status(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get training status for a specific result."""
        try:
            if result_id not in self.training_results:
                return None
            
            result = self.training_results[result_id]
            
            status = {
                "result_id": result_id,
                "model_id": result.model_id,
                "status": "completed" if result.end_time else "training",
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "success": result.success,
                "metrics": result.metrics,
                "error_message": result.error_message
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return None

    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get performance metrics for a model."""
        try:
            performance_data = {
                "model_id": model_id,
                "versions": [],
                "overall_performance": {},
                "deployment_history": []
            }
            
            # Get all versions for the model
            for version_id, version in self.model_versions.items():
                if version.model_id == model_id:
                    performance_data["versions"].append({
                        "version_id": version_id,
                        "version_number": version.version_number,
                        "performance_metrics": version.performance_metrics,
                        "deployment_status": version.deployment_status.value,
                        "created_at": version.created_at.isoformat(),
                        "deployed_at": version.deployed_at.isoformat() if version.deployed_at else None
                    })
            
            # Calculate overall performance
            if performance_data["versions"]:
                all_metrics = [v["performance_metrics"] for v in performance_data["versions"]]
                performance_data["overall_performance"] = {
                    "avg_accuracy": unified_math.mean([m.get("accuracy", 0) for m in all_metrics]),
                    "best_accuracy": max([m.get("accuracy", 0) for m in all_metrics]),
                    "total_versions": len(performance_data["versions"]),
                    "deployed_versions": sum(1 for v in performance_data["versions"] if v["deployment_status"] == "deployed")
                }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {"model_id": model_id, "error": str(e)}

    def get_trainer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trainer statistics."""
        total_configs = len(self.training_configs)
        total_results = len(self.training_results)
        total_versions = len(self.model_versions)
        active_models = len(self.active_models)
        
        # Calculate success rates
        successful_trainings = sum(1 for r in self.training_results.values() if r.success)
        success_rate = successful_trainings / total_results if total_results > 0 else 0.0
        
        # Calculate average performance
        all_metrics = [r.metrics for r in self.training_results.values() if r.success]
        avg_accuracy = unified_math.mean([m.get("accuracy", 0) for m in all_metrics]) if all_metrics else 0.0
        
        return {
            "total_configurations": total_configs,
            "total_training_results": total_results,
            "total_model_versions": total_versions,
            "active_models": active_models,
            "training_success_rate": success_rate,
            "average_accuracy": avg_accuracy,
            "training_queue_size": len(self.training_queue),
            "training_environments": len(self.training_environments)
        }

def main() -> None:
    """Main function for testing and demonstration."""
    trainer = AdaptiveTrainer("./test_adaptive_trainer_config.json")
    
    # Create a training configuration
    config_id = trainer.create_training_config(
        model_type="profit_predictor",
        training_mode=TrainingMode.BATCH,
        hyperparameters={"batch_size": 500, "epochs": 50}
    )
    
    safe_print(f"Created training configuration: {config_id}")
    
    # Get statistics
    stats = trainer.get_trainer_statistics()
    safe_print(f"Trainer Statistics: {stats}")

if __name__ == "__main__":
    main() 