import React from 'react';
import { Hash } from 'lucide-react';
import { useTrading } from '../contexts/TradingContext';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export function HashStreamPanel() {
  const { state } = useTrading();
  const { hashStream } = state;

  return (
    <div className="space-y-4">
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-bold mb-3 flex items-center">
          <Hash className="w-5 h-5 mr-2" />
          Live Hash Stream
        </h3>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {hashStream.slice(-10).map((hash, i) => (
            <div key={i} className="flex items-center space-x-4 text-sm bg-gray-700 p-2 rounded">
              <div className="font-mono text-blue-400">{hash.hash}</div>
              <div className="text-gray-400">E:{hash.entropy}</div>
              <div className={`text-xs px-2 py-1 rounded ${
                hash.confidence > 0.7 ? 'bg-green-900 text-green-300' : 'bg-yellow-900 text-yellow-300'
              }`}>
                {(hash.confidence * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-gray-500">
                {hash.pattern.slice(0, 4).join('-')}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-bold mb-3">Hash Entropy Distribution</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={hashStream.slice(-20).map((h, i) => ({ index: i, entropy: h.entropy }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="index" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="entropy" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
} 