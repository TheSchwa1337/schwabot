import React, { useState } from 'react';
import { Activity, Clock, Database } from 'lucide-react';
import { TradingProvider, useTrading } from './contexts/TradingContext';
import { ParadoxPanel } from './components/ParadoxPanel';
import { HashStreamPanel } from './components/HashStreamPanel';
import { RingValuesPanel } from './components/RingValuesPanel';
import { TradingSignalsPanel } from './components/TradingSignalsPanel';

function Dashboard() {
  const [activeMode, setActiveMode] = useState('paradox');
  const [isLive, setIsLive] = useState(false);
  const [detonationActive, setDetonationActive] = useState(false);
  const { state, dispatch, sendMessage } = useTrading();

  const handleDetonationTrigger = () => {
    setDetonationActive(true);
    sendMessage({
      type: 'DETONATION_TRIGGER',
      data: { timestamp: Date.now() },
      timestamp: Date.now()
    });
    setTimeout(() => setDetonationActive(false), 3000);
  };

  return (
    <div className="w-full h-screen bg-gradient-to-br from-black via-gray-900 to-blue-900 text-white overflow-hidden">
      {/* Header Controls */}
      <div className="flex justify-between items-center p-4 border-b border-gray-700">
        <div className="flex items-center space-x-4">
          <h1 className="text-xl font-bold">Schwabot Trading Dashboard</h1>
          <div className="flex space-x-2">
            {['paradox', 'hash', 'rings', 'signals'].map(mode => (
              <button
                key={mode}
                onClick={() => setActiveMode(mode)}
                className={`px-3 py-1 rounded text-sm ${
                  activeMode === mode ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-sm">
            <span className="text-gray-400">TPF State:</span>
            <span className={`ml-2 font-mono ${
              state.tpfState === 'TPF_STABILIZED' ? 'text-green-400' :
              state.tpfState === 'PARADOX_DETECTED' ? 'text-red-400' : 'text-yellow-400'
            }`}>
              {state.tpfState}
            </span>
          </div>
          
          <button
            onClick={() => setIsLive(!isLive)}
            className={`px-4 py-2 rounded font-bold ${
              isLive ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-600 hover:bg-gray-700'
            }`}
          >
            {isLive ? 'LIVE' : 'START'}
          </button>
          
          <button
            onClick={handleDetonationTrigger}
            className={`px-4 py-2 rounded font-bold ${
              detonationActive 
                ? 'bg-red-600 animate-pulse' 
                : 'bg-orange-600 hover:bg-orange-700'
            }`}
          >
            {detonationActive ? 'DETONATING...' : '1337 PROTOCOL'}
          </button>
        </div>
      </div>

      {/* Main Dashboard Content */}
      <div className="flex h-full">
        {/* Left Panel - Core Visualizations */}
        <div className="flex-1 p-4 space-y-4">
          {activeMode === 'paradox' && <ParadoxPanel />}
          {activeMode === 'hash' && <HashStreamPanel />}
          {activeMode === 'rings' && <RingValuesPanel />}
          {activeMode === 'signals' && <TradingSignalsPanel />}
        </div>

        {/* Right Panel - Status & Metrics */}
        <div className="w-80 p-4 border-l border-gray-700 space-y-4">
          {/* Market Status */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-bold mb-3 flex items-center">
              <Activity className="w-5 h-5 mr-2" />
              Market Status
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Price:</span>
                <span className="font-mono">${state.marketData.price.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span>Volume:</span>
                <span className="font-mono">{state.marketData.volume.toFixed(0)}</span>
              </div>
              <div className="flex justify-between">
                <span>RSI:</span>
                <span className={`font-mono ${
                  state.marketData.rsi > 70 ? 'text-red-400' : 
                  state.marketData.rsi < 30 ? 'text-green-400' : 'text-gray-300'
                }`}>
                  {state.marketData.rsi.toFixed(1)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>ATR:</span>
                <span className="font-mono">{state.marketData.atr.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span>Kelly:</span>
                <span className="font-mono">{state.marketData.kellyFraction.toFixed(3)}</span>
              </div>
            </div>
          </div>

          {/* System Metrics */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-bold mb-3 flex items-center">
              <Database className="w-5 h-5 mr-2" />
              System Metrics
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Hash Entropy:</span>
                <span className="font-mono">
                  {(state.hashStream.reduce((sum, h) => sum + h.entropy, 0) / Math.max(state.hashStream.length, 1)).toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Signal Strength:</span>
                <span className="font-mono">
                  {(state.glyphSignals.filter(s => s.confidence > 0.7).length / Math.max(state.glyphSignals.length, 1) * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span>Ring Stability:</span>
                <span className="font-mono">
                  {(Object.values(state.ringValues).reduce((sum, val) => sum + Math.abs(val), 0) / 10).toFixed(3)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>TPF Coherence:</span>
                <span className="font-mono">
                  {(state.stabilized ? 1.0 : state.paradoxVisible ? 0.3 : 0.6).toFixed(3)}
                </span>
              </div>
            </div>
          </div>

          {/* Recent Activity Log */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-bold mb-3 flex items-center">
              <Clock className="w-5 h-5 mr-2" />
              Activity Log
            </h3>
            <div className="space-y-1 text-xs max-h-48 overflow-y-auto">
              {state.timingHashes.slice(-10).map((entry, i) => (
                <div key={i} className="flex items-center space-x-2 text-gray-400">
                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                  <span>{new Date(entry.timestamp).toLocaleTimeString()}</span>
                  <span className="font-mono">{entry.hash.substring(0, 6)}...</span>
                  <span className={`text-xs px-1 rounded ${
                    entry.state === 'TPF_STABILIZED' ? 'bg-green-900 text-green-300' :
                    entry.state === 'PARADOX_DETECTED' ? 'bg-red-900 text-red-300' :
                    'bg-gray-700 text-gray-300'
                  }`}>
                    {entry.state}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <TradingProvider>
      <Dashboard />
    </TradingProvider>
  );
} 