import React, { useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface DREMState {
  entropy: number;
  stability: string;
  psi_magnitude: number;
  phi_magnitude: number;
  collapse_value: number;
  phase_state: string;
}

interface DREMVisualizerProps {
  state: DREMState;
  history: DREMState[];
}

const DREMVisualizer: React.FC<DREMVisualizerProps> = ({ state, history }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw field visualization
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;

    // Draw phase state circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, 100, 0, 2 * Math.PI);
    ctx.strokeStyle = state.stability === 'Stable' ? '#4CAF50' : '#F44336';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw collapse value indicator
    const collapseAngle = (state.collapse_value + 1) * Math.PI;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(
      centerX + 100 * Math.cos(collapseAngle),
      centerY + 100 * Math.sin(collapseAngle)
    );
    ctx.strokeStyle = '#2196F3';
    ctx.lineWidth = 3;
    ctx.stroke();
  }, [state]);

  const entropyData = {
    labels: history.map((_, i) => i.toString()),
    datasets: [
      {
        label: 'Entropy',
        data: history.map(h => h.entropy),
        borderColor: '#4CAF50',
        tension: 0.4,
      },
    ],
  };

  const magnitudeData = {
    labels: history.map((_, i) => i.toString()),
    datasets: [
      {
        label: 'Ψ Magnitude',
        data: history.map(h => h.psi_magnitude),
        borderColor: '#2196F3',
        tension: 0.4,
      },
      {
        label: 'Φ Magnitude',
        data: history.map(h => h.phi_magnitude),
        borderColor: '#F44336',
        tension: 0.4,
      },
    ],
  };

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">DREM Strategy Visualizer</h2>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-4 bg-gray-50 rounded">
          <h3 className="font-semibold mb-2">Current State</h3>
          <div className="space-y-2">
            <p>Entropy: {state.entropy.toFixed(3)}</p>
            <p>Stability: {state.stability}</p>
            <p>Phase State: {state.phase_state}</p>
            <p>Collapse Value: {state.collapse_value.toFixed(3)}</p>
          </div>
        </div>
        
        <div className="p-4 bg-gray-50 rounded">
          <h3 className="font-semibold mb-2">Field Magnitudes</h3>
          <div className="space-y-2">
            <p>Ψ Magnitude: {state.psi_magnitude.toFixed(3)}</p>
            <p>Φ Magnitude: {state.phi_magnitude.toFixed(3)}</p>
          </div>
        </div>
      </div>

      <div className="mb-4">
        <canvas
          ref={canvasRef}
          width={400}
          height={400}
          className="border border-gray-200 rounded"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <h3 className="font-semibold mb-2">Entropy History</h3>
          <Line data={entropyData} options={{
            responsive: true,
            scales: {
              y: {
                beginAtZero: true,
                max: 1
              }
            }
          }} />
        </div>
        
        <div>
          <h3 className="font-semibold mb-2">Field Magnitudes History</h3>
          <Line data={magnitudeData} options={{
            responsive: true,
            scales: {
              y: {
                beginAtZero: true
              }
            }
          }} />
        </div>
      </div>
    </div>
  );
};

export default DREMVisualizer; 