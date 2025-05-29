import React, { createContext, useContext, useReducer, useCallback } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

interface TradingState {
  marketData: {
    price: number;
    volume: number;
    rsi: number;
    entropy: number;
    drift: number;
    vwap: number;
    atr: number;
    kellyFraction: number;
  };
  ringValues: {
    R1: number;
    R2: number;
    R3: number;
    R4: number;
    R5: number;
    R6: number;
    R7: number;
    R8: number;
    R9: number;
    R10: number;
  };
  hashStream: Array<{
    timestamp: number;
    hash: string;
    entropy: number;
    confidence: number;
    pattern: number[];
  }>;
  timingHashes: Array<{
    timestamp: number;
    hash: string;
    state: string;
  }>;
  glyphSignals: Array<{
    timestamp: number;
    type: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    price: number;
    tpfState: string;
    hashTrigger: string;
  }>;
  tpfState: string;
  paradoxVisible: boolean;
  stabilized: boolean;
  phase: number;
}

type TradingAction =
  | { type: 'UPDATE_MARKET_DATA'; payload: Partial<TradingState['marketData']> }
  | { type: 'UPDATE_RING_VALUES'; payload: Partial<TradingState['ringValues']> }
  | { type: 'ADD_HASH'; payload: TradingState['hashStream'][0] }
  | { type: 'ADD_TIMING_HASH'; payload: TradingState['timingHashes'][0] }
  | { type: 'ADD_SIGNAL'; payload: TradingState['glyphSignals'][0] }
  | { type: 'UPDATE_TPF_STATE'; payload: { state: string; visible: boolean; stabilized: boolean; phase: number } };

const initialState: TradingState = {
  marketData: {
    price: 50000,
    volume: 1000,
    rsi: 50,
    entropy: 0.5,
    drift: 0,
    vwap: 49950,
    atr: 500,
    kellyFraction: 0.25
  },
  ringValues: {
    R1: 0, R2: 0, R3: 0, R4: 0, R5: 0,
    R6: 0, R7: 0, R8: 0, R9: 0, R10: 0
  },
  hashStream: [],
  timingHashes: [],
  glyphSignals: [],
  tpfState: 'INITIALIZING',
  paradoxVisible: false,
  stabilized: false,
  phase: 0
};

function tradingReducer(state: TradingState, action: TradingAction): TradingState {
  switch (action.type) {
    case 'UPDATE_MARKET_DATA':
      return {
        ...state,
        marketData: { ...state.marketData, ...action.payload }
      };
    case 'UPDATE_RING_VALUES':
      return {
        ...state,
        ringValues: { ...state.ringValues, ...action.payload }
      };
    case 'ADD_HASH':
      return {
        ...state,
        hashStream: [...state.hashStream.slice(-49), action.payload]
      };
    case 'ADD_TIMING_HASH':
      return {
        ...state,
        timingHashes: [...state.timingHashes.slice(-19), action.payload]
      };
    case 'ADD_SIGNAL':
      return {
        ...state,
        glyphSignals: [...state.glyphSignals.slice(-19), action.payload]
      };
    case 'UPDATE_TPF_STATE':
      return {
        ...state,
        tpfState: action.payload.state,
        paradoxVisible: action.payload.visible,
        stabilized: action.payload.stabilized,
        phase: action.payload.phase
      };
    default:
      return state;
  }
}

const TradingContext = createContext<{
  state: TradingState;
  dispatch: React.Dispatch<TradingAction>;
  sendMessage: (message: any) => void;
} | null>(null);

export function TradingProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(tradingReducer, initialState);
  const { sendMessage } = useWebSocket();

  const handleMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'MARKET_UPDATE':
        dispatch({ type: 'UPDATE_MARKET_DATA', payload: message.data });
        break;
      case 'RING_UPDATE':
        dispatch({ type: 'UPDATE_RING_VALUES', payload: message.data });
        break;
      case 'HASH_UPDATE':
        dispatch({ type: 'ADD_HASH', payload: message.data });
        break;
      case 'TIMING_UPDATE':
        dispatch({ type: 'ADD_TIMING_HASH', payload: message.data });
        break;
      case 'SIGNAL_UPDATE':
        dispatch({ type: 'ADD_SIGNAL', payload: message.data });
        break;
      case 'TPF_UPDATE':
        dispatch({ type: 'UPDATE_TPF_STATE', payload: message.data });
        break;
    }
  }, []);

  return (
    <TradingContext.Provider value={{ state, dispatch, sendMessage }}>
      {children}
    </TradingContext.Provider>
  );
}

export function useTrading() {
  const context = useContext(TradingContext);
  if (!context) {
    throw new Error('useTrading must be used within a TradingProvider');
  }
  return context;
} 