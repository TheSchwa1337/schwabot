import React, { useState, useEffect, useCallback } from 'react';
import { Circle, Square, Triangle, TrendingUp, TrendingDown, Activity } from 'lucide-react';

interface ParadoxVisualizerProps {
  data: {
    state: {
      phase: number;
      stabilized: boolean;
      paradox_visible: boolean;
      trading_mode: boolean;
      glyph_state: string;
      detonation_protocol: boolean;
    };
    market_data: {
      price: number;
      volume: number;
      rsi: number;
      drift: number;
      entropy: number;
    };
    trading_signals: Array<{
      id: number;
      type: string;
      confidence: number;
      price: number;
      timestamp: number;
    }>;
    tpf_metrics: {
      magnitude: number;
      phase: number;
      stability_score: number;
      paradox_intensity: number;
      detonation_ready: boolean;
    };
  };
  onDetonationTrigger: () => void;
}

export default function ParadoxVisualizer({ data, onDetonationTrigger }: ParadoxVisualizerProps) {
  const [showGEMM, setShowGEMM] = useState(false);
  const [showStopBook, setShowStopBook] = useState(false);
  const [showMathDeltas, setShowMathDeltas] = useState(false);

  return (
    <div className="flex flex-col items-center justify-center w-full h-full p-6 bg-gradient-to-br from-black via-gray-900 to-blue-900 text-white">
      <div className="flex justify-between w-full max-w-6xl mb-6">
        <div className="flex flex-col space-y-2">
          <h2 className="text-xl font-bold">
            {data.state.stabilized ? 
              "TPF Stabilized - Paradox Resolved" : 
              data.state.paradox_visible ? 
                "TPF Active - Processing Paradox" : 
                "Recursive System Initialization"}
          </h2>
          <div className="text-sm text-gray-300">
            Phase: {data.state.phase} | Glyph State: {data.state.glyph_state}
          </div>
        </div>
        
        <div className="flex space-x-4">
          <button 
            onClick={() => setShowGEMM(!showGEMM)}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded"
          >
            {showGEMM ? 'Hide GEMM' : 'Show GEMM'}
          </button>
          <button 
            onClick={() => setShowStopBook(!showStopBook)}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded"
          >
            {showStopBook ? 'Hide StopBook' : 'Show StopBook'}
          </button>
          <button 
            onClick={() => setShowMathDeltas(!showMathDeltas)}
            className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded"
          >
            {showMathDeltas ? 'Hide MathLib' : 'Show MathLib'}
          </button>
          <button 
            onClick={onDetonationTrigger}
            className={`px-4 py-2 rounded font-bold ${
              data.state.detonation_protocol 
                ? 'bg-red-600 animate-pulse' 
                : 'bg-orange-600 hover:bg-orange-700'
            }`}
          >
            {data.state.detonation_protocol ? 'DETONATING...' : '1337 PROTOCOL'}
          </button>
        </div>
      </div>

      <div className="flex space-x-8 w-full max-w-6xl">
        {/* Main Paradox Visualization */}
        <div className="flex-1">
          <div className="relative w-64 h-64 mb-8 mx-auto">
            {/* Base triangle visualization */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className={`text-green-400 transform transition-all duration-500 ${
                data.state.paradox_visible ? 'scale-110' : 'scale-100'
              } ${data.state.detonation_protocol ? 'animate-spin' : ''}`}>
                <svg viewBox="0 0 100 100" width="240" height="240">
                  {/* Standard Triangle */}
                  <polygon 
                    points="50,10 10,90 90,90" 
                    fill="none" 
                    stroke={data.state.stabilized ? "#00ff00" : data.state.paradox_visible ? "#ff3300" : "#00aa00"} 
                    strokeWidth="1.5"
                  />
                  
                  {/* The "invisible" fourth side */}
                  <line 
                    x1="10" y1="90" 
                    x2="90" y2="90" 
                    stroke={data.state.paradox_visible && !data.state.stabilized ? "#ff3300" : "#00aa00"} 
                    strokeWidth="1.5" 
                    strokeDasharray={data.state.paradox_visible && !data.state.stabilized ? "2,2" : "0"} 
                    opacity={data.state.paradox_visible ? "1" : "0.2"}
                  />
                  
                  {/* Inner circle - representing the recursive core */}
                  <circle 
                    cx="50" 
                    cy="60" 
                    r="25" 
                    fill="none" 
                    stroke={data.state.stabilized ? "#00ff00" : "#666666"} 
                    strokeWidth="0.75" 
                    opacity={data.state.paradox_visible ? "1" : "0.3"}
                  />
                  
                  {/* Market data integration - price oscillation */}
                  {data.state.trading_mode && (
                    <circle 
                      cx="50" 
                      cy="60" 
                      r={15 + (data.market_data.rsi / 100) * 10} 
                      fill="none" 
                      stroke="#00aaff" 
                      strokeWidth="0.5" 
                      opacity="0.6"
                    />
                  )}
                  
                  {/* GEMM Overlay */}
                  {showGEMM && data.state.stabilized && (
                    <>
                      <circle cx="50" cy="60" r="8" stroke="#FF00FF" strokeWidth="0.5" />
                      <polygon points="40,40 60,40 50,70" stroke="#AA00FF" fill="none" strokeWidth="0.5" />
                    </>
                  )}
                  
                  {/* StopBook Overlay */}
                  {showStopBook && data.state.paradox_visible && (
                    <>
                      {data.trading_signals.map((signal, idx) => (
                        <circle
                          key={signal.id}
                          cx={50 + Math.sin(idx) * 20}
                          cy={60 + Math.cos(idx) * 20}
                          r={3}
                          fill={
                            signal.type === "SELL" ? "#FF0000" :
                            signal.type === "BUY" ? "#00FF00" :
                            "#FFFF00"
                          }
                          opacity="0.6"
                        />
                      ))}
                    </>
                  )}
                  
                  {/* MathLib Delta Overlay */}
                  {showMathDeltas && (
                    <circle
                      cx="50"
                      cy="60"
                      r={15 + data.market_data.rsi / 10}
                      stroke="#00AACC"
                      strokeWidth="0.5"
                      fill="none"
                      opacity="0.4"
                    />
                  )}
                </svg>
              </div>
            </div>
            
            {/* Paradox effect - the rippling effect when paradox is detected */}
            {data.state.paradox_visible && !data.state.stabilized && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-64 h-64 rounded-full border border-red-500 opacity-30 animate-ping transition-all duration-300">
                </div>
              </div>
            )}
            
            {/* TPF stabilization wave effect */}
            {data.state.stabilized && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-64 h-64 rounded-full border border-green-400 opacity-20 animate-pulse transition-all duration-1000">
                </div>
              </div>
            )}

            {/* Detonation Protocol Visual Effect */}
            {data.state.detonation_protocol && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-80 h-80 rounded-full border-4 border-orange-500 animate-ping opacity-70">
                </div>
                <div className="absolute w-72 h-72 rounded-full border-2 border-red-500 animate-pulse opacity-50">
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Trading Data Panel */}
        {data.state.trading_mode && (
          <div className="w-80 space-y-4">
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-lg font-bold mb-3 flex items-center">
                <Activity className="w-5 h-5 mr-2" />
                Market Data
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Price:</span>
                  <span className="font-mono">${data.market_data.price.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Volume:</span>
                  <span className="font-mono">{data.market_data.volume.toFixed(0)}</span>
                </div>
                <div className="flex justify-between">
                  <span>RSI:</span>
                  <span className={`font-mono ${
                    data.market_data.rsi > 70 ? 'text-red-400' : 
                    data.market_data.rsi < 30 ? 'text-green-400' : 'text-gray-300'
                  }`}>
                    {data.market_data.rsi.toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Drift:</span>
                  <span className="font-mono">{data.market_data.drift.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Entropy:</span>
                  <span className="font-mono">{data.market_data.entropy.toFixed(3)}</span>
                </div>
              </div>
            </div>

            {/* Trading Signals */}
            {data.trading_signals.length > 0 && (
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-3">Trading Signals</h3>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {data.trading_signals.map(signal => (
                    <div 
                      key={signal.id}
                      className={`p-2 rounded text-xs flex justify-between items-center ${
                        signal.type === 'BUY' ? 'bg-green-900 text-green-300' :
                        signal.type === 'SELL' ? 'bg-red-900 text-red-300' :
                        'bg-yellow-900 text-yellow-300'
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        {signal.type === 'BUY' ? <TrendingUp className="w-3 h-3" /> :
                         signal.type === 'SELL' ? <TrendingDown className="w-3 h-3" /> :
                         <Activity className="w-3 h-3" />}
                        <span className="font-bold">{signal.type}</span>
                      </div>
                      <div className="text-right">
                        <div>${signal.price.toFixed(2)}</div>
                        <div className="text-xs opacity-75">
                          {(signal.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Status Information */}
      <div className="flex flex-col space-y-2 max-w-4xl w-full">
        <div className="px-4 py-2 bg-gray-800 rounded-lg text-sm">
          <p className="font-mono">
            {data.state.paradox_visible ? 
              data.state.stabilized ? 
                <span className="text-green-400">TPF Fractal: Paradox integrated into recursive framework</span> :
                <span className="text-red-400">Paradox Detected: Triangle with 3 sides contains 4 points</span> :
              <span className="text-gray-400">System initializing recursive geometric analysis...</span>
            }
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="px-4 py-2 bg-gray-800 rounded-lg text-sm flex justify-between">
            <span className="font-mono">Status:</span>
            <span className={`font-mono ${
              data.state.stabilized ? 'text-green-400' : 
              data.state.paradox_visible ? 'text-red-400' : 'text-gray-400'
            }`}>
              {data.state.stabilized ? 'STABLE' : data.state.paradox_visible ? 'PARADOX DETECTED' : 'INITIALIZING'}
            </span>
          </div>
          
          <div className="px-4 py-2 bg-gray-800 rounded-lg text-sm flex justify-between">
            <span className="font-mono">TPF Module:</span>
            <span className={`font-mono ${
              data.state.stabilized ? 'text-green-400' : 
              data.state.paradox_visible ? 'text-yellow-400' : 'text-gray-400'
            }`}>
              {data.state.stabilized ? 'ACTIVE - PARADOX RESOLVED' : 
               data.state.paradox_visible ? 'PROCESSING' : 'STANDBY'}
            </span>
          </div>

          <div className="px-4 py-2 bg-gray-800 rounded-lg text-sm flex justify-between">
            <span className="font-mono">Trading Mode:</span>
            <span className={`font-mono ${data.state.trading_mode ? 'text-blue-400' : 'text-gray-400'}`}>
              {data.state.trading_mode ? 'ACTIVE' : 'STANDBY'}
            </span>
          </div>
        </div>
      </div>
      
      <div className="mt-6 text-center text-xs text-gray-500">
        <p>TPF: The Paradox Fractals - Recursive System Visualization</p>
        <p>Recursive Quantum AI Klein Bottle System + Trading Integration</p>
        <p>1337 P4TT3RN D3T0N4T10N PR0T0C0L Ready</p>
      </div>
    </div>
  );
} 