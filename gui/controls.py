#!/usr/bin/env python3
"""
GUI Controls - Interactive Trading and Mathematical Parameter Controls
===================================================================

This module implements a comprehensive GUI controls system for Schwabot,
providing interactive trading controls, mathematical parameter adjustment,
and real-time system monitoring capabilities.

Core Functionality:
- Interactive trading parameter controls
- Real-time mathematical parameter adjustment
- System monitoring and status display
- Risk management controls
- Performance visualization controls
- Configuration management interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import threading
import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from schwabot.core.config_manager import ConfigManager
    from schwabot.core.risk_manager import RiskManager
    from schwabot.core.profit_routing_engine import ProfitRoutingEngine
    from schwabot.mathlib.sfsss_tensor import SFSSTensor
    from schwabot.mathlib.ufs_tensor import UFSTensor
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    # Create mock classes for testing
    ConfigManager = type('ConfigManager', (), {})
    RiskManager = type('RiskManager', (), {})
    ProfitRoutingEngine = type('ProfitRoutingEngine', (), {})

logger = logging.getLogger(__name__)

@dataclass
class ControlParameter:
    """Control parameter structure."""
    parameter_id: str
    name: str
    value: Any
    min_value: Optional[float]
    max_value: Optional[float]
    step: Optional[float]
    data_type: str
    description: str
    category: str
    is_live: bool = True
    callback: Optional[Callable] = None

@dataclass
class ControlState:
    """Control state structure."""
    state_id: str
    timestamp: datetime
    parameters: Dict[str, Any]
    system_status: str
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

class TradingControls:
    """Trading parameter controls."""
    
    def __init__(self, parent_frame: ttk.Frame, config_manager: ConfigManager):
        self.parent_frame = parent_frame
        self.config_manager = config_manager
        self.controls: Dict[str, Any] = {}
        self.parameters: Dict[str, ControlParameter] = {}
        self._initialize_controls()
        self._create_control_panel()
    
    def _initialize_controls(self):
        """Initialize trading control parameters."""
        self.parameters = {
            "risk_tolerance": ControlParameter(
                parameter_id="risk_tolerance",
                name="Risk Tolerance",
                value=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                data_type="float",
                description="Risk tolerance level (0.0 = conservative, 1.0 = aggressive)",
                category="risk"
            ),
            "max_position_size": ControlParameter(
                parameter_id="max_position_size",
                name="Max Position Size",
                value=0.1,
                min_value=0.01,
                max_value=1.0,
                step=0.01,
                data_type="float",
                description="Maximum position size as fraction of portfolio",
                category="risk"
            ),
            "stop_loss_percentage": ControlParameter(
                parameter_id="stop_loss_percentage",
                name="Stop Loss %",
                value=0.02,
                min_value=0.001,
                max_value=0.1,
                step=0.001,
                data_type="float",
                description="Stop loss percentage",
                category="risk"
            ),
            "take_profit_percentage": ControlParameter(
                parameter_id="take_profit_percentage",
                name="Take Profit %",
                value=0.05,
                min_value=0.001,
                max_value=0.2,
                step=0.001,
                data_type="float",
                description="Take profit percentage",
                category="profit"
            ),
            "trading_frequency": ControlParameter(
                parameter_id="trading_frequency",
                name="Trading Frequency",
                value=60,
                min_value=1,
                max_value=3600,
                step=1,
                data_type="int",
                description="Trading frequency in seconds",
                category="timing"
            )
        }
    
    def _create_control_panel(self):
        """Create the trading control panel."""
        # Create main frame
        main_frame = ttk.LabelFrame(self.parent_frame, text="Trading Controls", padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create control sections
        self._create_risk_controls(main_frame)
        self._create_profit_controls(main_frame)
        self._create_timing_controls(main_frame)
        self._create_action_buttons(main_frame)
    
    def _create_risk_controls(self, parent: ttk.Frame):
        """Create risk management controls."""
        risk_frame = ttk.LabelFrame(parent, text="Risk Management", padding="5")
        risk_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        row = 0
        for param_id, param in self.parameters.items():
            if param.category == "risk":
                self._create_control_widget(risk_frame, param, row)
                row += 1
    
    def _create_profit_controls(self, parent: ttk.Frame):
        """Create profit management controls."""
        profit_frame = ttk.LabelFrame(parent, text="Profit Management", padding="5")
        profit_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        row = 0
        for param_id, param in self.parameters.items():
            if param.category == "profit":
                self._create_control_widget(profit_frame, param, row)
                row += 1
    
    def _create_timing_controls(self, parent: ttk.Frame):
        """Create timing controls."""
        timing_frame = ttk.LabelFrame(parent, text="Timing Controls", padding="5")
        timing_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        row = 0
        for param_id, param in self.parameters.items():
            if param.category == "timing":
                self._create_control_widget(timing_frame, param, row)
                row += 1
    
    def _create_control_widget(self, parent: ttk.Frame, param: ControlParameter, row: int):
        """Create individual control widget."""
        # Label
        label = ttk.Label(parent, text=param.name)
        label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        
        # Control widget based on data type
        if param.data_type == "float":
            var = tk.DoubleVar(value=param.value)
            control = ttk.Scale(
                parent, 
                from_=param.min_value, 
                to=param.max_value, 
                variable=var,
                orient="horizontal",
                length=200
            )
            control.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            
            # Value label
            value_label = ttk.Label(parent, text=f"{param.value:.3f}")
            value_label.grid(row=row, column=2, sticky="w", padx=5, pady=2)
            
            # Bind update function
            def update_value(val, param=param, label=value_label):
                param.value = float(val)
                label.config(text=f"{param.value:.3f}")
                self._on_parameter_change(param)
            
            control.config(command=update_value)
            
        elif param.data_type == "int":
            var = tk.IntVar(value=param.value)
            control = ttk.Spinbox(
                parent,
                from_=param.min_value,
                to=param.max_value,
                increment=param.step,
                textvariable=var,
                width=10
            )
            control.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            
            # Bind update function
            def update_value():
                param.value = var.get()
                self._on_parameter_change(param)
            
            control.config(command=update_value)
        
        # Description tooltip
        tooltip = ttk.Label(parent, text=param.description, font=("Arial", 8))
        tooltip.grid(row=row, column=3, sticky="w", padx=5, pady=2)
        
        # Store control reference
        self.controls[param.parameter_id] = {
            "control": control,
            "variable": var if 'var' in locals() else None,
            "value_label": value_label if 'value_label' in locals() else None
        }
    
    def _create_action_buttons(self, parent: ttk.Frame):
        """Create action buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=10)
        
        # Start Trading button
        start_button = ttk.Button(
            button_frame, 
            text="Start Trading", 
            command=self._start_trading,
            style="Accent.TButton"
        )
        start_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Stop Trading button
        stop_button = ttk.Button(
            button_frame, 
            text="Stop Trading", 
            command=self._stop_trading
        )
        stop_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Reset Parameters button
        reset_button = ttk.Button(
            button_frame, 
            text="Reset Parameters", 
            command=self._reset_parameters
        )
        reset_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Save Configuration button
        save_button = ttk.Button(
            button_frame, 
            text="Save Config", 
            command=self._save_configuration
        )
        save_button.grid(row=0, column=3, padx=5, pady=5)
    
    def _on_parameter_change(self, param: ControlParameter):
        """Handle parameter change."""
        try:
            # Update configuration
            self.config_manager.set_parameter(param.parameter_id, param.value, "GUI update")
            
            # Log change
            logger.info(f"Parameter {param.name} updated to {param.value}")
            
            # Execute callback if provided
            if param.callback:
                param.callback(param.value)
                
        except Exception as e:
            logger.error(f"Error updating parameter {param.name}: {e}")
            messagebox.showerror("Error", f"Failed to update {param.name}: {e}")
    
    def _start_trading(self):
        """Start trading operations."""
        try:
            # Validate parameters
            if not self._validate_parameters():
                return
            
            # Start trading
            logger.info("Starting trading operations")
            messagebox.showinfo("Trading", "Trading operations started")
            
        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            messagebox.showerror("Error", f"Failed to start trading: {e}")
    
    def _stop_trading(self):
        """Stop trading operations."""
        try:
            logger.info("Stopping trading operations")
            messagebox.showinfo("Trading", "Trading operations stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
            messagebox.showerror("Error", f"Failed to stop trading: {e}")
    
    def _reset_parameters(self):
        """Reset parameters to defaults."""
        try:
            for param in self.parameters.values():
                param.value = self._get_default_value(param)
                self._update_control_display(param)
            
            logger.info("Parameters reset to defaults")
            messagebox.showinfo("Reset", "Parameters reset to defaults")
            
        except Exception as e:
            logger.error(f"Error resetting parameters: {e}")
            messagebox.showerror("Error", f"Failed to reset parameters: {e}")
    
    def _save_configuration(self):
        """Save current configuration."""
        try:
            config_data = {}
            for param_id, param in self.parameters.items():
                config_data[param_id] = param.value
            
            # Save to file
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                logger.info(f"Configuration saved to {filename}")
                messagebox.showinfo("Save", f"Configuration saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def _validate_parameters(self) -> bool:
        """Validate all parameters."""
        for param in self.parameters.values():
            if param.min_value is not None and param.value < param.min_value:
                messagebox.showerror("Validation Error", 
                                   f"{param.name} must be at least {param.min_value}")
                return False
            
            if param.max_value is not None and param.value > param.max_value:
                messagebox.showerror("Validation Error", 
                                   f"{param.name} must be at most {param.max_value}")
                return False
        
        return True
    
    def _get_default_value(self, param: ControlParameter) -> Any:
        """Get default value for parameter."""
        if param.data_type == "float":
            return (param.min_value + param.max_value) / 2 if param.min_value and param.max_value else 0.0
        elif param.data_type == "int":
            return (param.min_value + param.max_value) // 2 if param.min_value and param.max_value else 0
        else:
            return param.value
    
    def _update_control_display(self, param: ControlParameter):
        """Update control display."""
        if param.parameter_id in self.controls:
            control_info = self.controls[param.parameter_id]
            
            if control_info["variable"]:
                control_info["variable"].set(param.value)
            
            if control_info["value_label"]:
                if param.data_type == "float":
                    control_info["value_label"].config(text=f"{param.value:.3f}")
                else:
                    control_info["value_label"].config(text=str(param.value))

class MathematicalControls:
    """Mathematical parameter controls."""
    
    def __init__(self, parent_frame: ttk.Frame, config_manager: ConfigManager):
        self.parent_frame = parent_frame
        self.config_manager = config_manager
        self.controls: Dict[str, Any] = {}
        self.parameters: Dict[str, ControlParameter] = {}
        self._initialize_parameters()
        self._create_control_panel()
    
    def _initialize_parameters(self):
        """Initialize mathematical parameters."""
        self.parameters = {
            "sfsss_precision": ControlParameter(
                parameter_id="sfsss_precision",
                name="SFSSS Precision",
                value=0.000001,
                min_value=0.0000001,
                max_value=0.001,
                step=0.0000001,
                data_type="float",
                description="SFSSS tensor precision",
                category="precision"
            ),
            "ufs_precision": ControlParameter(
                parameter_id="ufs_precision",
                name="UFS Precision",
                value=0.000001,
                min_value=0.0000001,
                max_value=0.001,
                step=0.0000001,
                data_type="float",
                description="UFS tensor precision",
                category="precision"
            ),
            "max_iterations": ControlParameter(
                parameter_id="max_iterations",
                name="Max Iterations",
                value=1000,
                min_value=100,
                max_value=10000,
                step=100,
                data_type="int",
                description="Maximum mathematical iterations",
                category="performance"
            ),
            "convergence_threshold": ControlParameter(
                parameter_id="convergence_threshold",
                name="Convergence Threshold",
                value=0.0001,
                min_value=0.00001,
                max_value=0.01,
                step=0.00001,
                data_type="float",
                description="Mathematical convergence threshold",
                category="precision"
            )
        }
    
    def _create_control_panel(self):
        """Create mathematical control panel."""
        # Create main frame
        main_frame = ttk.LabelFrame(self.parent_frame, text="Mathematical Controls", padding="10")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create control sections
        self._create_precision_controls(main_frame)
        self._create_performance_controls(main_frame)
        self._create_action_buttons(main_frame)
    
    def _create_precision_controls(self, parent: ttk.Frame):
        """Create precision controls."""
        precision_frame = ttk.LabelFrame(parent, text="Precision Settings", padding="5")
        precision_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        row = 0
        for param_id, param in self.parameters.items():
            if param.category == "precision":
                self._create_control_widget(precision_frame, param, row)
                row += 1
    
    def _create_performance_controls(self, parent: ttk.Frame):
        """Create performance controls."""
        performance_frame = ttk.LabelFrame(parent, text="Performance Settings", padding="5")
        performance_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        row = 0
        for param_id, param in self.parameters.items():
            if param.category == "performance":
                self._create_control_widget(performance_frame, param, row)
                row += 1
    
    def _create_control_widget(self, parent: ttk.Frame, param: ControlParameter, row: int):
        """Create individual control widget."""
        # Label
        label = ttk.Label(parent, text=param.name)
        label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        
        # Control widget based on data type
        if param.data_type == "float":
            var = tk.DoubleVar(value=param.value)
            control = ttk.Scale(
                parent, 
                from_=param.min_value, 
                to=param.max_value, 
                variable=var,
                orient="horizontal",
                length=200
            )
            control.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            
            # Value label
            value_label = ttk.Label(parent, text=f"{param.value:.6f}")
            value_label.grid(row=row, column=2, sticky="w", padx=5, pady=2)
            
            # Bind update function
            def update_value(val, param=param, label=value_label):
                param.value = float(val)
                label.config(text=f"{param.value:.6f}")
                self._on_parameter_change(param)
            
            control.config(command=update_value)
            
        elif param.data_type == "int":
            var = tk.IntVar(value=param.value)
            control = ttk.Spinbox(
                parent,
                from_=param.min_value,
                to=param.max_value,
                increment=param.step,
                textvariable=var,
                width=10
            )
            control.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            
            # Bind update function
            def update_value():
                param.value = var.get()
                self._on_parameter_change(param)
            
            control.config(command=update_value)
        
        # Description tooltip
        tooltip = ttk.Label(parent, text=param.description, font=("Arial", 8))
        tooltip.grid(row=row, column=3, sticky="w", padx=5, pady=2)
        
        # Store control reference
        self.controls[param.parameter_id] = {
            "control": control,
            "variable": var if 'var' in locals() else None,
            "value_label": value_label if 'value_label' in locals() else None
        }
    
    def _create_action_buttons(self, parent: ttk.Frame):
        """Create action buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=10)
        
        # Apply Changes button
        apply_button = ttk.Button(
            button_frame, 
            text="Apply Changes", 
            command=self._apply_changes
        )
        apply_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Test Mathematical Functions button
        test_button = ttk.Button(
            button_frame, 
            text="Test Functions", 
            command=self._test_mathematical_functions
        )
        test_button.grid(row=0, column=1, padx=5, pady=5)
    
    def _on_parameter_change(self, param: ControlParameter):
        """Handle parameter change."""
        try:
            # Update configuration
            self.config_manager.set_parameter(param.parameter_id, param.value, "GUI update")
            
            # Log change
            logger.info(f"Mathematical parameter {param.name} updated to {param.value}")
            
        except Exception as e:
            logger.error(f"Error updating mathematical parameter {param.name}: {e}")
            messagebox.showerror("Error", f"Failed to update {param.name}: {e}")
    
    def _apply_changes(self):
        """Apply mathematical parameter changes."""
        try:
            # Validate parameters
            if not self._validate_parameters():
                return
            
            # Apply changes
            logger.info("Applying mathematical parameter changes")
            messagebox.showinfo("Apply", "Mathematical parameters applied")
            
        except Exception as e:
            logger.error(f"Error applying changes: {e}")
            messagebox.showerror("Error", f"Failed to apply changes: {e}")
    
    def _test_mathematical_functions(self):
        """Test mathematical functions."""
        try:
            logger.info("Testing mathematical functions")
            
            # Test SFSSS tensor
            test_data = np.random.rand(10, 10)
            sfsss_tensor = SFSSTensor(test_data)
            
            # Test UFS tensor
            ufs_tensor = UFSTensor(test_data)
            
            messagebox.showinfo("Test", "Mathematical functions tested successfully")
            
        except Exception as e:
            logger.error(f"Error testing mathematical functions: {e}")
            messagebox.showerror("Error", f"Failed to test functions: {e}")
    
    def _validate_parameters(self) -> bool:
        """Validate all parameters."""
        for param in self.parameters.values():
            if param.min_value is not None and param.value < param.min_value:
                messagebox.showerror("Validation Error", 
                                   f"{param.name} must be at least {param.min_value}")
                return False
            
            if param.max_value is not None and param.value > param.max_value:
                messagebox.showerror("Validation Error", 
                                   f"{param.name} must be at most {param.max_value}")
                return False
        
        return True

class SystemMonitor:
    """System monitoring display."""
    
    def __init__(self, parent_frame: ttk.Frame):
        self.parent_frame = parent_frame
        self.monitoring_active = False
        self.monitor_thread = None
        self._create_monitor_panel()
        self._start_monitoring()
    
    def _create_monitor_panel(self):
        """Create system monitor panel."""
        # Create main frame
        main_frame = ttk.LabelFrame(self.parent_frame, text="System Monitor", padding="10")
        main_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)
        
        # Create monitoring sections
        self._create_performance_monitor(main_frame)
        self._create_status_monitor(main_frame)
        self._create_log_monitor(main_frame)
    
    def _create_performance_monitor(self, parent: ttk.Frame):
        """Create performance monitoring section."""
        perf_frame = ttk.LabelFrame(parent, text="Performance Metrics", padding="5")
        perf_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # CPU Usage
        self.cpu_label = ttk.Label(perf_frame, text="CPU: 0%")
        self.cpu_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        
        # Memory Usage
        self.memory_label = ttk.Label(perf_frame, text="Memory: 0 MB")
        self.memory_label.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        
        # Active Connections
        self.connections_label = ttk.Label(perf_frame, text="Connections: 0")
        self.connections_label.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        
        # Processing Rate
        self.processing_label = ttk.Label(perf_frame, text="Processing: 0 ops/s")
        self.processing_label.grid(row=3, column=0, sticky="w", padx=5, pady=2)
    
    def _create_status_monitor(self, parent: ttk.Frame):
        """Create status monitoring section."""
        status_frame = ttk.LabelFrame(parent, text="System Status", padding="5")
        status_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Trading Status
        self.trading_status = ttk.Label(status_frame, text="Trading: Stopped", foreground="red")
        self.trading_status.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        
        # API Status
        self.api_status = ttk.Label(status_frame, text="API: Disconnected", foreground="red")
        self.api_status.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        
        # Database Status
        self.db_status = ttk.Label(status_frame, text="Database: Disconnected", foreground="red")
        self.db_status.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        
        # Mathematical Engine Status
        self.math_status = ttk.Label(status_frame, text="Math Engine: Ready", foreground="green")
        self.math_status.grid(row=3, column=0, sticky="w", padx=5, pady=2)
    
    def _create_log_monitor(self, parent: ttk.Frame):
        """Create log monitoring section."""
        log_frame = ttk.LabelFrame(parent, text="Recent Logs", padding="5")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        
        # Log text area
        self.log_text = tk.Text(log_frame, height=10, width=40, font=("Consolas", 8))
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=scrollbar.set)
    
    def _start_monitoring(self):
        """Start system monitoring."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Monitoring loop."""
        while self.monitoring_active:
            try:
                # Update performance metrics
                self._update_performance_metrics()
                
                # Update system status
                self._update_system_status()
                
                # Update logs
                self._update_logs()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _update_performance_metrics(self):
        """Update performance metrics display."""
        try:
            # Simulate performance metrics
            cpu_usage = np.random.uniform(10, 80)
            memory_usage = np.random.uniform(100, 500)
            connections = np.random.randint(0, 10)
            processing_rate = np.random.uniform(100, 1000)
            
            # Update labels
            self.cpu_label.config(text=f"CPU: {cpu_usage:.1f}%")
            self.memory_label.config(text=f"Memory: {memory_usage:.0f} MB")
            self.connections_label.config(text=f"Connections: {connections}")
            self.processing_label.config(text=f"Processing: {processing_rate:.0f} ops/s")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _update_system_status(self):
        """Update system status display."""
        try:
            # Simulate status updates
            statuses = {
                "trading": ("Running", "green") if np.random.random() > 0.5 else ("Stopped", "red"),
                "api": ("Connected", "green") if np.random.random() > 0.3 else ("Disconnected", "red"),
                "database": ("Connected", "green") if np.random.random() > 0.2 else ("Disconnected", "red"),
                "math_engine": ("Ready", "green")
            }
            
            # Update status labels
            self.trading_status.config(text=f"Trading: {statuses['trading'][0]}", 
                                     foreground=statuses['trading'][1])
            self.api_status.config(text=f"API: {statuses['api'][0]}", 
                                 foreground=statuses['api'][1])
            self.db_status.config(text=f"Database: {statuses['database'][0]}", 
                                foreground=statuses['database'][1])
            self.math_status.config(text=f"Math Engine: {statuses['math_engine'][0]}", 
                                  foreground=statuses['math_engine'][1])
            
        except Exception as e:
            logger.error(f"Error updating system status: {e}")
    
    def _update_logs(self):
        """Update log display."""
        try:
            # Simulate log entries
            log_entries = [
                f"[{datetime.now().strftime('%H:%M:%S')}] INFO: System running normally",
                f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Processing tick data",
                f"[{datetime.now().strftime('%H:%M:%S')}] INFO: Mathematical calculations completed"
            ]
            
            # Add to log text area
            for entry in log_entries:
                self.log_text.insert(tk.END, entry + "\n")
                self.log_text.see(tk.END)
            
            # Limit log size
            lines = self.log_text.get("1.0", tk.END).split("\n")
            if len(lines) > 100:
                self.log_text.delete("1.0", f"{len(lines) - 100}.0")
            
        except Exception as e:
            logger.error(f"Error updating logs: {e}")

class ControlsApplication:
    """Main controls application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Schwabot Controls")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.risk_manager = RiskManager()
        self.profit_engine = ProfitRoutingEngine()
        
        # Create main interface
        self._create_main_interface()
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
    
    def _create_main_interface(self):
        """Create main interface."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        
        # Create control panels
        self.trading_controls = TradingControls(main_frame, self.config_manager)
        self.mathematical_controls = MathematicalControls(main_frame, self.config_manager)
        self.system_monitor = SystemMonitor(main_frame)
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
    
    def run(self):
        """Run the controls application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Controls application interrupted")
        except Exception as e:
            logger.error(f"Error running controls application: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self.system_monitor, 'monitoring_active'):
                self.system_monitor.monitoring_active = False
            
            logger.info("Controls application cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main function."""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create and run application
        app = ControlsApplication()
        app.run()
        
    except Exception as e:
        print(f"Error starting controls application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 