import React from 'react';
import { Target, TrendingUp, TrendingDown, Activity } from 'lucide-react';
import { useTrading } from '../contexts/TradingContext';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts';

export function TradingSignalsPanel() {
  const { state } = useTrading();
  const { glyphSignals } = state;

  return (
    <div className="space-y-4">
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-bold mb-3 flex items-center">
          <Target className="w-5 h-5 mr-2" />
          Glyph Trading Signals
        </h3>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {glyphSignals.slice(-8).map((signal, i) => (
            <div key={i} className={`p-3 rounded flex justify-between items-center ${
              signal.type === 'BUY' ? 'bg-green-900' :
              signal.type === 'SELL' ? 'bg-red-900' : 'bg-yellow-900'
            }`}>
              <div className="flex items-center space-x-3">
                {signal.type === 'BUY' ? <TrendingUp className="w-4 h-4" /> :
                 signal.type === 'SELL' ? <TrendingDown className="w-4 h-4" /> :
                 <Activity className="w-4 h-4" />}
                <span className="font-bold">{signal.type}</span>
                <span className="text-sm">${signal.price.toFixed(2)}</span>
              </div>
              <div className="text-right text-sm">
                <div>{(signal.confidence * 100).toFixed(0)}%</div>
                <div className="text-xs text-gray-400">{signal.tpfState}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-bold mb-3">Signal Confidence Over Time</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={glyphSignals.slice(-20).map((s, i) => ({ index: i, confidence: s.confidence * 100 }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="index" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Line type="monotone" dataKey="confidence" stroke="#8884d8" dot={false} />
              <ReferenceLine y={70} stroke="#ff6b6b" strokeDasharray="2 2" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
} 