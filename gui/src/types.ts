export interface VisualizerState {
  phase: number;
  stabilized: boolean;
  paradox_visible: boolean;
  trading_mode: boolean;
  glyph_state: string;
  detonation_protocol: boolean;
}

export interface MarketData {
  price: number;
  volume: number;
  rsi: number;
  drift: number;
  entropy: number;
}

export interface TradingSignal {
  id: number;
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  timestamp: number;
}

export interface TPFMetrics {
  magnitude: number;
  phase: number;
  stability_score: number;
  paradox_intensity: number;
  detonation_ready: boolean;
}

export interface VisualizerData {
  state: VisualizerState;
  market_data: MarketData;
  trading_signals: TradingSignal[];
  tpf_metrics: TPFMetrics;
} 