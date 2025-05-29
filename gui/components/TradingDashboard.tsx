import React, { useState, useEffect } from 'react';
import DREMVisualizer from './DREMVisualizer';

interface TradingDashboardProps {
  // ... existing props ...
}

const TradingDashboard: React.FC<TradingDashboardProps> = (props) => {
  // ... existing state ...
  const [dremState, setDremState] = useState({
    entropy: 0,
    stability: "Unknown",
    psi_magnitude: 0,
    phi_magnitude: 0,
    collapse_value: 0,
    phase_state: "Initial"
  });
  const [dremHistory, setDremHistory] = useState<any[]>([]);

  useEffect(() => {
    // Subscribe to DREM strategy updates
    const dremSubscription = props.strategyManager.subscribeToDREM((state) => {
      setDremState(state);
      setDremHistory(prev => [...prev, state].slice(-50)); // Keep last 50 states
    });

    return () => {
      dremSubscription.unsubscribe();
    };
  }, [props.strategyManager]);

  return (
    <div className="p-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Existing dashboard components */}
        <div className="col-span-1">
          {/* ... existing components ... */}
        </div>

        {/* DREM Strategy Section */}
        <div className="col-span-1">
          <DREMVisualizer state={dremState} history={dremHistory} />
          
          <div className="mt-4 p-4 bg-white rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-2">DREM Strategy Controls</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Field Dimensions
                </label>
                <input
                  type="number"
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                  value={props.strategyManager.getDREMDimensions()}
                  onChange={(e) => props.strategyManager.setDREMDimensions(parseInt(e.target.value))}
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Entropy Threshold
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  className="mt-1 block w-full"
                  value={props.strategyManager.getDREMEntropyThreshold()}
                  onChange={(e) => props.strategyManager.setDREMEntropyThreshold(parseFloat(e.target.value))}
                />
              </div>

              <div className="flex space-x-4">
                <button
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  onClick={() => props.strategyManager.resetDREM()}
                >
                  Reset DREM
                </button>
                
                <button
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                  onClick={() => props.strategyManager.toggleDREM()}
                >
                  {props.strategyManager.isDREMEnabled() ? 'Disable DREM' : 'Enable DREM'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingDashboard; 