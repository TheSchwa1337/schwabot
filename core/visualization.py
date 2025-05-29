"""
Visualization module for Schwabot System
Provides plotting and analysis functions for trading data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

def plot_profit_decay(
    times: np.ndarray,
    profits: np.ndarray,
    base_value: float = 0.618,
    curve_factor: float = 0.777,
    title: str = "Profit Decay Analysis",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot profit decay curve with actual data.
    If save_path is provided, saves the plot to that path, else shows the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual data
    ax.scatter(times, profits, label='Actual Profits', alpha=0.6)
    
    # Plot theoretical decay curve
    theoretical = base_value * np.exp(-curve_factor * times)
    ax.plot(times, theoretical, 'r--', label='Theoretical Decay')
    
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Profit')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        plt.show()
    
    return fig, ax

def plot_tick_sequence(
    ticks: np.ndarray,
    window_size: int = 5,
    title: str = "Tick Sequence Analysis",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot tick sequence with trend detection.
    If save_path is provided, saves the plot to that path, else shows the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot tick values
    ax.plot(ticks, label='Tick Values', alpha=0.7)
    
    # Calculate and plot moving average
    ma = pd.Series(ticks).rolling(window=window_size).mean()
    ax.plot(ma, 'r--', label=f'{window_size}-Tick MA')
    
    # Add trend zones
    trend = np.mean(np.gradient(ticks[-window_size:]))
    if trend > 0.002:
        ax.axhspan(ticks.min(), ticks.max(), alpha=0.2, color='g', label='Uptrend')
    elif trend < -0.002:
        ax.axhspan(ticks.min(), ticks.max(), alpha=0.2, color='r', label='Downtrend')
    else:
        ax.axhspan(ticks.min(), ticks.max(), alpha=0.2, color='y', label='Neutral')
    
    ax.set_title(title)
    ax.set_xlabel('Tick Index')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        plt.show()
    
    return fig, ax

def plot_hash_distribution(
    hashes: List[str],
    title: str = "Hash Distribution Analysis",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot hash value distribution.
    If save_path is provided, saves the plot to that path, else shows the plot.
    """
    # Convert hashes to numeric values
    values = [int(h[:8], 16) for h in hashes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(values, bins=50, ax=ax)
    
    # Add KDE
    sns.kdeplot(values, ax=ax, color='r')
    
    ax.set_title(title)
    ax.set_xlabel('Hash Value (First 8 Hex Digits)')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        plt.show()
    
    return fig, ax

def plot_trade_metrics(
    trades: List[Dict[str, Any]],
    title: str = "Trade Performance Analysis",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot trade performance metrics.
    If save_path is provided, saves the plot to that path, else shows the plot.
    """
    # Convert trades to DataFrame
    df = pd.DataFrame(trades)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot profit distribution
    sns.histplot(data=df, x='profit', bins=30, ax=ax)
    
    # Add mean line
    ax.axvline(df['profit'].mean(), color='r', linestyle='--', 
               label=f'Mean: {df["profit"].mean():.2f}')
    
    ax.set_title(title)
    ax.set_xlabel('Profit')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        plt.show()
    
    return fig, ax

def create_interactive_dashboard(
    data: pd.DataFrame,
    title: str = "Schwabot Trading Dashboard"
) -> go.Figure:
    """Create interactive Plotly dashboard (no save_path, as Plotly handles saving differently)."""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    ))
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['volume'],
        name='Volume',
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        xaxis_rangeslider_visible=False
    )
    
    return fig

def plot_system_metrics(
    metrics: Dict[str, List[float]],
    timestamps: List[datetime],
    title: str = "System Performance Metrics",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot system performance metrics.
    If save_path is provided, saves the plot to that path, else shows the plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot CPU metrics
    ax1.plot(timestamps, metrics['cpu_temp'], 'r-', label='CPU Temp')
    ax1.plot(timestamps, metrics['cpu_load'], 'b-', label='CPU Load')
    ax1.set_ylabel('CPU Metrics')
    ax1.legend()
    ax1.grid(True)
    
    # Plot GPU metrics
    ax2.plot(timestamps, metrics['gpu_temp'], 'r-', label='GPU Temp')
    ax2.plot(timestamps, metrics['gpu_load'], 'b-', label='GPU Load')
    ax2.set_ylabel('GPU Metrics')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        plt.show()
    
    return fig, (ax1, ax2) 