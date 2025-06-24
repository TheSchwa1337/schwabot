#!/usr/bin/env python3
"""
Data Visualizer - Real-time Charts and Mathematical Visualization System
=====================================================================

This module implements a comprehensive data visualization system for Schwabot,
providing real-time charts, mathematical plots, and interactive dashboards
for trading data analysis and system monitoring.

Core Functionality:
- Real-time price charts and technical indicators
- Mathematical tensor visualization
- System performance dashboards
- Interactive trading analysis tools
- 3D mathematical surface plots
- Real-time data streaming visualization
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import logging
import os
import sys
from collections import deque

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from schwabot.core.multi_bit_btc_processor import MultiBitBTCProcessor
    from schwabot.mathlib.sfsss_tensor import SFSSTensor
    from schwabot.mathlib.ufs_tensor import UFSTensor
    from schwabot.core.profit_routing_engine import ProfitRoutingEngine
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    # Create mock classes for testing
    MultiBitBTCProcessor = type('MultiBitBTCProcessor', (), {})
    SFSSTensor = type('SFSSTensor', (), {})
    UFSTensor = type('UFSTensor', (), {})
    ProfitRoutingEngine = type('ProfitRoutingEngine', (), {})

logger = logging.getLogger(__name__)

@dataclass
class ChartData:
    """Chart data structure."""
    chart_id: str
    data_type: str
    timestamps: List[datetime]
    values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    config_id: str
    chart_type: str
    update_interval: float
    data_points: int
    colors: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class PriceChartVisualizer:
    """Real-time price chart visualizer."""
    
    def __init__(self, parent_frame: ttk.Frame):
        self.parent_frame = parent_frame
        self.data_buffer = deque(maxlen=1000)
        self.is_live = False
        self.update_thread = None
        self._create_chart_panel()
        self._initialize_chart()
    
    def _create_chart_panel(self):
        """Create chart control panel."""
        # Create main frame
        main_frame = ttk.LabelFrame(self.parent_frame, text="Price Chart", padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create control buttons
        self._create_control_buttons(main_frame)
        
        # Create chart area
        self._create_chart_area(main_frame)
        
        # Configure grid weights
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
    
    def _create_control_buttons(self, parent: ttk.Frame):
        """Create control buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Start/Stop Live button
        self.live_button = ttk.Button(
            button_frame,
            text="Start Live",
            command=self._toggle_live_mode
        )
        self.live_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Clear Chart button
        clear_button = ttk.Button(
            button_frame,
            text="Clear Chart",
            command=self._clear_chart
        )
        clear_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Save Chart button
        save_button = ttk.Button(
            button_frame,
            text="Save Chart",
            command=self._save_chart
        )
        save_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Chart type selector
        ttk.Label(button_frame, text="Chart Type:").grid(row=0, column=3, padx=5, pady=5)
        self.chart_type_var = tk.StringVar(value="line")
        chart_type_combo = ttk.Combobox(
            button_frame,
            textvariable=self.chart_type_var,
            values=["line", "candlestick", "bar", "area"],
            state="readonly",
            width=10
        )
        chart_type_combo.grid(row=0, column=4, padx=5, pady=5)
        chart_type_combo.bind("<<ComboboxSelected>>", self._on_chart_type_change)
    
    def _create_chart_area(self, parent: ttk.Frame):
        """Create chart display area."""
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.get_tk_widget().grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # Create toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Initialize empty chart
        self._initialize_chart()
    
    def _initialize_chart(self):
        """Initialize empty chart."""
        self.ax.clear()
        self.ax.set_title("Real-time Price Chart")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.grid(True, alpha=0.3)
        
        # Set initial data
        self.ax.plot([], [], 'b-', linewidth=1, label="Price")
        self.ax.legend()
        
        self.canvas.draw()
    
    def _toggle_live_mode(self):
        """Toggle live data mode."""
        if self.is_live:
            self._stop_live_mode()
        else:
            self._start_live_mode()
    
    def _start_live_mode(self):
        """Start live data mode."""
        self.is_live = True
        self.live_button.config(text="Stop Live")
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._live_update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Live chart mode started")
    
    def _stop_live_mode(self):
        """Stop live data mode."""
        self.is_live = False
        self.live_button.config(text="Start Live")
        
        logger.info("Live chart mode stopped")
    
    def _live_update_loop(self):
        """Live update loop."""
        while self.is_live:
            try:
                # Generate simulated price data
                self._generate_simulated_data()
                
                # Update chart
                self._update_chart()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in live update loop: {e}")
                time.sleep(5)
    
    def _generate_simulated_data(self):
        """Generate simulated price data."""
        try:
            # Simulate BTC price movement
            base_price = 50000.0
            price_change = np.random.normal(0, 100)
            current_price = base_price + price_change
            
            # Add to data buffer
            self.data_buffer.append({
                "timestamp": datetime.now(),
                "price": current_price,
                "volume": np.random.uniform(0.1, 10.0)
            })
            
        except Exception as e:
            logger.error(f"Error generating simulated data: {e}")
    
    def _update_chart(self):
        """Update chart display."""
        try:
            if not self.data_buffer:
                return
            
            # Extract data
            timestamps = [item["timestamp"] for item in self.data_buffer]
            prices = [item["price"] for item in self.data_buffer]
            
            # Clear and redraw
            self.ax.clear()
            self.ax.set_title("Real-time Price Chart")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price")
            self.ax.grid(True, alpha=0.3)
            
            # Plot based on chart type
            chart_type = self.chart_type_var.get()
            if chart_type == "line":
                self.ax.plot(timestamps, prices, 'b-', linewidth=1, label="Price")
            elif chart_type == "bar":
                self.ax.bar(range(len(prices)), prices, alpha=0.7, label="Price")
            elif chart_type == "area":
                self.ax.fill_between(range(len(prices)), prices, alpha=0.3, label="Price")
            else:  # candlestick
                self.ax.plot(timestamps, prices, 'b-', linewidth=1, label="Price")
            
            self.ax.legend()
            
            # Rotate x-axis labels for better readability
            plt.setp(self.ax.get_xticklabels(), rotation=45)
            
            # Update canvas
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating chart: {e}")
    
    def _clear_chart(self):
        """Clear chart data."""
        try:
            self.data_buffer.clear()
            self._initialize_chart()
            logger.info("Chart cleared")
            
        except Exception as e:
            logger.error(f"Error clearing chart: {e}")
    
    def _save_chart(self):
        """Save chart to file."""
        try:
            filename = f"price_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Save", f"Chart saved as {filename}")
            logger.info(f"Chart saved as {filename}")
            
        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            messagebox.showerror("Error", f"Failed to save chart: {e}")
    
    def _on_chart_type_change(self, event):
        """Handle chart type change."""
        try:
            self._update_chart()
        except Exception as e:
            logger.error(f"Error changing chart type: {e}")

class MathematicalVisualizer:
    """Mathematical tensor and surface visualizer."""
    
    def __init__(self, parent_frame: ttk.Frame):
        self.parent_frame = parent_frame
        self.current_tensor = None
        self._create_visualizer_panel()
        self._initialize_3d_plot()
    
    def _create_visualizer_panel(self):
        """Create mathematical visualizer panel."""
        # Create main frame
        main_frame = ttk.LabelFrame(self.parent_frame, text="Mathematical Visualization", padding="10")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create control panel
        self._create_control_panel(main_frame)
        
        # Create 3D plot area
        self._create_3d_plot_area(main_frame)
        
        # Configure grid weights
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
    
    def _create_control_panel(self, parent: ttk.Frame):
        """Create control panel."""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Tensor type selector
        ttk.Label(control_frame, text="Tensor Type:").grid(row=0, column=0, padx=5, pady=5)
        self.tensor_type_var = tk.StringVar(value="sfsss")
        tensor_combo = ttk.Combobox(
            control_frame,
            textvariable=self.tensor_type_var,
            values=["sfsss", "ufs", "random"],
            state="readonly",
            width=10
        )
        tensor_combo.grid(row=0, column=1, padx=5, pady=5)
        tensor_combo.bind("<<ComboboxSelected>>", self._on_tensor_type_change)
        
        # Generate button
        generate_button = ttk.Button(
            control_frame,
            text="Generate Tensor",
            command=self._generate_tensor
        )
        generate_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Plot type selector
        ttk.Label(control_frame, text="Plot Type:").grid(row=0, column=3, padx=5, pady=5)
        self.plot_type_var = tk.StringVar(value="surface")
        plot_combo = ttk.Combobox(
            control_frame,
            textvariable=self.plot_type_var,
            values=["surface", "wireframe", "contour"],
            state="readonly",
            width=10
        )
        plot_combo.grid(row=0, column=4, padx=5, pady=5)
        plot_combo.bind("<<ComboboxSelected>>", self._on_plot_type_change)
        
        # Save button
        save_button = ttk.Button(
            control_frame,
            text="Save Plot",
            command=self._save_plot
        )
        save_button.grid(row=0, column=5, padx=5, pady=5)
    
    def _create_3d_plot_area(self, parent: ttk.Frame):
        """Create 3D plot display area."""
        # Create matplotlib figure with 3D projection
        self.figure_3d = Figure(figsize=(12, 8), dpi=100)
        self.ax_3d = self.figure_3d.add_subplot(111, projection='3d')
        
        # Create canvas
        self.canvas_3d = FigureCanvasTkAgg(self.figure_3d, parent)
        self.canvas_3d.get_tk_widget().grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # Create toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        toolbar = NavigationToolbar2Tk(self.canvas_3d, toolbar_frame)
        toolbar.update()
    
    def _initialize_3d_plot(self):
        """Initialize 3D plot."""
        self.ax_3d.clear()
        self.ax_3d.set_title("Mathematical Tensor Visualization")
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        
        self.canvas_3d.draw()
    
    def _on_tensor_type_change(self, event):
        """Handle tensor type change."""
        try:
            self._generate_tensor()
        except Exception as e:
            logger.error(f"Error changing tensor type: {e}")
    
    def _on_plot_type_change(self, event):
        """Handle plot type change."""
        try:
            if self.current_tensor is not None:
                self._plot_tensor()
        except Exception as e:
            logger.error(f"Error changing plot type: {e}")
    
    def _generate_tensor(self):
        """Generate mathematical tensor."""
        try:
            tensor_type = self.tensor_type_var.get()
            
            if tensor_type == "sfsss":
                # Generate SFSSS tensor
                data = np.random.rand(20, 20)
                self.current_tensor = SFSSTensor(data)
            elif tensor_type == "ufs":
                # Generate UFS tensor
                data = np.random.rand(20, 20)
                self.current_tensor = UFSTensor(data)
            else:  # random
                # Generate random tensor
                self.current_tensor = np.random.rand(20, 20)
            
            self._plot_tensor()
            logger.info(f"Generated {tensor_type} tensor")
            
        except Exception as e:
            logger.error(f"Error generating tensor: {e}")
    
    def _plot_tensor(self):
        """Plot tensor data."""
        try:
            if self.current_tensor is None:
                return
            
            # Extract data
            if hasattr(self.current_tensor, 'data'):
                data = self.current_tensor.data
            else:
                data = self.current_tensor
            
            # Create coordinate grids
            x = np.arange(data.shape[0])
            y = np.arange(data.shape[1])
            X, Y = np.meshgrid(x, y)
            
            # Clear plot
            self.ax_3d.clear()
            self.ax_3d.set_title("Mathematical Tensor Visualization")
            self.ax_3d.set_xlabel("X")
            self.ax_3d.set_ylabel("Y")
            self.ax_3d.set_zlabel("Z")
            
            # Plot based on type
            plot_type = self.plot_type_var.get()
            if plot_type == "surface":
                self.ax_3d.plot_surface(X, Y, data, cmap='viridis', alpha=0.8)
            elif plot_type == "wireframe":
                self.ax_3d.plot_wireframe(X, Y, data, color='blue', alpha=0.8)
            else:  # contour
                self.ax_3d.contour(X, Y, data, levels=20, cmap='viridis')
            
            self.canvas_3d.draw()
            
        except Exception as e:
            logger.error(f"Error plotting tensor: {e}")
    
    def _save_plot(self):
        """Save 3D plot to file."""
        try:
            filename = f"tensor_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.figure_3d.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Save", f"Plot saved as {filename}")
            logger.info(f"Plot saved as {filename}")
            
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            messagebox.showerror("Error", f"Failed to save plot: {e}")

class PerformanceDashboard:
    """System performance dashboard."""
    
    def __init__(self, parent_frame: ttk.Frame):
        self.parent_frame = parent_frame
        self.metrics_data = deque(maxlen=100)
        self.is_monitoring = False
        self.monitor_thread = None
        self._create_dashboard_panel()
        self._initialize_dashboard()
    
    def _create_dashboard_panel(self):
        """Create dashboard panel."""
        # Create main frame
        main_frame = ttk.LabelFrame(self.parent_frame, text="Performance Dashboard", padding="10")
        main_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)
        
        # Create control buttons
        self._create_dashboard_controls(main_frame)
        
        # Create metrics display
        self._create_metrics_display(main_frame)
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
    
    def _create_dashboard_controls(self, parent: ttk.Frame):
        """Create dashboard controls."""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Start/Stop Monitoring button
        self.monitor_button = ttk.Button(
            control_frame,
            text="Start Monitoring",
            command=self._toggle_monitoring
        )
        self.monitor_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Refresh button
        refresh_button = ttk.Button(
            control_frame,
            text="Refresh",
            command=self._refresh_dashboard
        )
        refresh_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Export button
        export_button = ttk.Button(
            control_frame,
            text="Export Data",
            command=self._export_data
        )
        export_button.grid(row=0, column=2, padx=5, pady=5)
    
    def _create_metrics_display(self, parent: ttk.Frame):
        """Create metrics display."""
        # Create notebook for different metric views
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Real-time metrics tab
        self._create_realtime_tab()
        
        # Historical metrics tab
        self._create_historical_tab()
        
        # System status tab
        self._create_status_tab()
    
    def _create_realtime_tab(self):
        """Create real-time metrics tab."""
        realtime_frame = ttk.Frame(self.notebook)
        self.notebook.add(realtime_frame, text="Real-time")
        
        # Create metrics labels
        self.cpu_label = ttk.Label(realtime_frame, text="CPU Usage: 0%", font=("Arial", 12))
        self.cpu_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        self.memory_label = ttk.Label(realtime_frame, text="Memory Usage: 0 MB", font=("Arial", 12))
        self.memory_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        
        self.processing_label = ttk.Label(realtime_frame, text="Processing Rate: 0 ops/s", font=("Arial", 12))
        self.processing_label.grid(row=2, column=0, sticky="w", padx=10, pady=5)
        
        self.accuracy_label = ttk.Label(realtime_frame, text="Prediction Accuracy: 0%", font=("Arial", 12))
        self.accuracy_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)
        
        # Create mini charts
        self._create_mini_charts(realtime_frame)
    
    def _create_historical_tab(self):
        """Create historical metrics tab."""
        historical_frame = ttk.Frame(self.notebook)
        self.notebook.add(historical_frame, text="Historical")
        
        # Create historical chart
        self.figure_hist = Figure(figsize=(8, 6), dpi=100)
        self.ax_hist = self.figure_hist.add_subplot(111)
        
        self.canvas_hist = FigureCanvasTkAgg(self.figure_hist, historical_frame)
        self.canvas_hist.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        historical_frame.grid_columnconfigure(0, weight=1)
        historical_frame.grid_rowconfigure(0, weight=1)
    
    def _create_status_tab(self):
        """Create system status tab."""
        status_frame = ttk.Frame(self.notebook)
        self.notebook.add(status_frame, text="Status")
        
        # System status indicators
        self.trading_status = ttk.Label(status_frame, text="Trading: Stopped", foreground="red", font=("Arial", 12))
        self.trading_status.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        self.api_status = ttk.Label(status_frame, text="API: Disconnected", foreground="red", font=("Arial", 12))
        self.api_status.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        
        self.database_status = ttk.Label(status_frame, text="Database: Disconnected", foreground="red", font=("Arial", 12))
        self.database_status.grid(row=2, column=0, sticky="w", padx=10, pady=5)
        
        self.math_engine_status = ttk.Label(status_frame, text="Math Engine: Ready", foreground="green", font=("Arial", 12))
        self.math_engine_status.grid(row=3, column=0, sticky="w", padx=10, pady=5)
    
    def _create_mini_charts(self, parent: ttk.Frame):
        """Create mini performance charts."""
        # Create mini chart for CPU usage
        cpu_frame = ttk.LabelFrame(parent, text="CPU Usage Trend")
        cpu_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        
        self.figure_cpu = Figure(figsize=(6, 3), dpi=100)
        self.ax_cpu = self.figure_cpu.add_subplot(111)
        
        self.canvas_cpu = FigureCanvasTkAgg(self.figure_cpu, cpu_frame)
        self.canvas_cpu.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        cpu_frame.grid_columnconfigure(0, weight=1)
        cpu_frame.grid_rowconfigure(0, weight=1)
    
    def _initialize_dashboard(self):
        """Initialize dashboard."""
        self._update_metrics_display()
        self._update_mini_charts()
    
    def _toggle_monitoring(self):
        """Toggle monitoring mode."""
        if self.is_monitoring:
            self._stop_monitoring()
        else:
            self._start_monitoring()
    
    def _start_monitoring(self):
        """Start performance monitoring."""
        self.is_monitoring = True
        self.monitor_button.config(text="Stop Monitoring")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def _stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        self.monitor_button.config(text="Start Monitoring")
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Monitoring loop."""
        while self.is_monitoring:
            try:
                # Generate simulated metrics
                self._generate_metrics()
                
                # Update display
                self._update_metrics_display()
                self._update_mini_charts()
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _generate_metrics(self):
        """Generate simulated performance metrics."""
        try:
            metrics = {
                "timestamp": datetime.now(),
                "cpu_usage": np.random.uniform(10, 80),
                "memory_usage": np.random.uniform(100, 500),
                "processing_rate": np.random.uniform(100, 1000),
                "prediction_accuracy": np.random.uniform(70, 95)
            }
            
            self.metrics_data.append(metrics)
            
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
    
    def _update_metrics_display(self):
        """Update metrics display."""
        try:
            if not self.metrics_data:
                return
            
            latest = self.metrics_data[-1]
            
            # Update labels
            self.cpu_label.config(text=f"CPU Usage: {latest['cpu_usage']:.1f}%")
            self.memory_label.config(text=f"Memory Usage: {latest['memory_usage']:.0f} MB")
            self.processing_label.config(text=f"Processing Rate: {latest['processing_rate']:.0f} ops/s")
            self.accuracy_label.config(text=f"Prediction Accuracy: {latest['prediction_accuracy']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error updating metrics display: {e}")
    
    def _update_mini_charts(self):
        """Update mini charts."""
        try:
            if len(self.metrics_data) < 2:
                return
            
            # Update CPU chart
            timestamps = [m["timestamp"] for m in self.metrics_data]
            cpu_values = [m["cpu_usage"] for m in self.metrics_data]
            
            self.ax_cpu.clear()
            self.ax_cpu.plot(timestamps, cpu_values, 'b-', linewidth=1)
            self.ax_cpu.set_title("CPU Usage Trend")
            self.ax_cpu.set_ylabel("CPU %")
            self.ax_cpu.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(self.ax_cpu.get_xticklabels(), rotation=45)
            
            self.canvas_cpu.draw()
            
        except Exception as e:
            logger.error(f"Error updating mini charts: {e}")
    
    def _refresh_dashboard(self):
        """Refresh dashboard."""
        try:
            self._update_metrics_display()
            self._update_mini_charts()
            logger.info("Dashboard refreshed")
            
        except Exception as e:
            logger.error(f"Error refreshing dashboard: {e}")
    
    def _export_data(self):
        """Export metrics data."""
        try:
            if not self.metrics_data:
                messagebox.showwarning("Export", "No data to export")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(list(self.metrics_data))
            
            # Save to CSV
            filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            
            messagebox.showinfo("Export", f"Data exported to {filename}")
            logger.info(f"Metrics data exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            messagebox.showerror("Error", f"Failed to export data: {e}")

class VisualizerApplication:
    """Main visualizer application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Schwabot Data Visualizer")
        self.root.geometry("1600x1000")
        
        # Initialize components
        self.btc_processor = MultiBitBTCProcessor()
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
        
        # Create visualization components
        self.price_chart = PriceChartVisualizer(main_frame)
        self.math_visualizer = MathematicalVisualizer(main_frame)
        self.performance_dashboard = PerformanceDashboard(main_frame)
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
    
    def run(self):
        """Run the visualizer application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Visualizer application interrupted")
        except Exception as e:
            logger.error(f"Error running visualizer application: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        try:
            # Stop monitoring
            if hasattr(self.performance_dashboard, 'is_monitoring'):
                self.performance_dashboard.is_monitoring = False
            
            if hasattr(self.price_chart, 'is_live'):
                self.price_chart.is_live = False
            
            logger.info("Visualizer application cleaned up")
            
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
        app = VisualizerApplication()
        app.run()
        
    except Exception as e:
        print(f"Error starting visualizer application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 