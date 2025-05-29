import React from 'react';
import { Brain } from 'lucide-react';
import { useTrading } from '../contexts/TradingContext';

export function ParadoxPanel() {
  const { state } = useTrading();
  const { marketData, tpfState, paradoxVisible, stabilized, phase } = state;

  return (
    <div className="space-y-4">
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-bold mb-3 flex items-center">
          <Brain className="w-5 h-5 mr-2" />
          Recursive Paradox Engine
        </h3>
        <div className="relative h-64 flex items-center justify-center">
          <svg viewBox="0 0 100 100" width="200" height="200">
            {/* Triangle */}
            <polygon 
              points="50,10 10,90 90,90" 
              fill="none" 
              stroke={stabilized ? "#00ff00" : paradoxVisible ? "#ff3300" : "#00aa00"} 
              strokeWidth="1.5"
            />
            {/* Fourth point paradox */}
            <circle 
              cx="50" 
              cy="90" 
              r="2" 
              fill="#ff3300" 
              opacity={paradoxVisible && !stabilized ? "1" : "0"}
            />
            {/* Inner circle */}
            <circle 
              cx="50" 
              cy="60" 
              r={20 + (marketData.rsi / 100) * 10} 
              fill="none" 
              stroke="#00aaff" 
              strokeWidth="0.5" 
              opacity="0.6"
            />
          </svg>
        </div>
        <div className="text-center text-sm text-gray-400">
          Phase: {phase} | State: {tpfState}
        </div>
      </div>
    </div>
  );
} 