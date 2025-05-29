import React from 'react';
import { Layers } from 'lucide-react';
import { useTrading } from '../contexts/TradingContext';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts';

export function RingValuesPanel() {
  const { state } = useTrading();
  const { ringValues } = state;

  return (
    <div className="space-y-4">
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-bold mb-3 flex items-center">
          <Layers className="w-5 h-5 mr-2" />
          RITTLE-GEMM Ring Values
        </h3>
        <div className="grid grid-cols-5 gap-4">
          {Object.entries(ringValues).map(([ring, value]) => (
            <div key={ring} className="text-center">
              <div className="text-xs text-gray-400">{ring}</div>
              <div className={`text-sm font-mono ${
                Math.abs(value) > 0.5 ? 'text-red-400' : 'text-green-400'
              }`}>
                {value.toFixed(3)}
              </div>
              <div className="w-full bg-gray-700 rounded h-2 mt-1">
                <div 
                  className="bg-blue-500 h-2 rounded transition-all duration-200"
                  style={{ width: `${Math.min(Math.abs(value) * 100, 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-bold mb-3">Ring Stability Over Time</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart data={Object.entries(ringValues).map(([ring, value], i) => ({ ring: i, value, name: ring }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="ring" />
              <YAxis />
              <Tooltip formatter={(value, name, props) => [value.toFixed(3), props.payload.name]} />
              <Scatter dataKey="value" fill="#8884d8" />
              <ReferenceLine y={0} stroke="#666" strokeDasharray="2 2" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
} 