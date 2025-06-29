#!/usr/bin/env python3
"""
Profit Cycle Allocator Module
=============================

Multi-stage, recursive profit allocation for Schwabot v0.05.
Provides intelligent profit distribution and reinvestment strategies.
"""

import time
import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class AllocationStage(Enum):
    """Allocation stage enumeration."""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    RESERVE = "reserve"


class AllocationType(Enum):
    """Allocation type enumeration."""
    REINVEST = "reinvest"
    WITHDRAW = "withdraw"
    RESERVE = "reserve"
    DISTRIBUTE = "distribute"
    COMPOUND = "compound"


@dataclass
class ProfitAllocation:
    """Profit allocation configuration."""
    allocation_id: str
    stage: AllocationStage
    allocation_type: AllocationType
    percentage: float  # 0.0 to 1.0
    min_amount: float
    max_amount: float
    priority: int
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AllocationResult:
    """Allocation execution result."""
    allocation_id: str
    timestamp: float
    stage: AllocationStage
    allocation_type: AllocationType
    amount: float
    percentage: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfitCycle:
    """Profit cycle data."""
    cycle_id: str
    start_time: float
    end_time: Optional[float] = None
    total_profit: float = 0.0
    allocated_profit: float = 0.0
    remaining_profit: float = 0.0
    allocations: List[AllocationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProfitCycleAllocator:
    """
    Profit Cycle Allocator for Schwabot v0.05.
    
    Provides multi-stage, recursive profit allocation with intelligent
    distribution and reinvestment strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the profit cycle allocator."""
        self.config = config or self._default_config()
        
        # Allocation management
        self.allocations: Dict[str, ProfitAllocation] = {}
        self.stage_allocations: Dict[AllocationStage, List[ProfitAllocation]] = {}
        
        # Cycle tracking
        self.current_cycle: Optional[ProfitCycle] = None
        self.cycle_history: List[ProfitCycle] = []
        self.max_history_size = self.config.get('max_history_size', 100)
        
        # Performance tracking
        self.total_cycles = 0
        self.total_profit = 0.0
        self.total_allocated = 0.0
        self.successful_allocations = 0
        self.failed_allocations = 0
        
        # State management
        self.last_update = time.time()
        
        # Initialize default allocations
        self._initialize_default_allocations()
        
        logger.info("💰 Profit Cycle Allocator initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'max_history_size': 100,
            'min_profit_threshold': 10.0,  # Minimum profit to trigger allocation
            'max_allocation_percentage': 0.95,  # Maximum 95% allocation
            'reserve_percentage': 0.05,  # Always keep 5% in reserve
            'compound_threshold': 100.0,  # Minimum amount for compounding
            'stage_priorities': {
                'immediate': 1,
                'short_term': 2,
                'medium_term': 3,
                'long_term': 4,
                'reserve': 5
            },
            'auto_compound_enabled': True,
            'risk_adjusted_allocation': True
        }
    
    def _initialize_default_allocations(self):
        """Initialize default profit allocations."""
        # Immediate allocations (high priority)
        self.add_allocation(
            "immediate_reinvest",
            AllocationStage.IMMEDIATE,
            AllocationType.REINVEST,
            0.30,  # 30%
            10.0,
            1000.0,
            1
        )
        
        self.add_allocation(
            "immediate_reserve",
            AllocationStage.IMMEDIATE,
            AllocationType.RESERVE,
            0.10,  # 10%
            5.0,
            500.0,
            2
        )
        
        # Short-term allocations
        self.add_allocation(
            "short_term_compound",
            AllocationStage.SHORT_TERM,
            AllocationType.COMPOUND,
            0.25,  # 25%
            25.0,
            2500.0,
            3
        )
        
        self.add_allocation(
            "short_term_distribute",
            AllocationStage.SHORT_TERM,
            AllocationType.DISTRIBUTE,
            0.15,  # 15%
            10.0,
            1000.0,
            4
        )
        
        # Medium-term allocations
        self.add_allocation(
            "medium_term_reinvest",
            AllocationStage.MEDIUM_TERM,
            AllocationType.REINVEST,
            0.15,  # 15%
            50.0,
            5000.0,
            5
        )
        
        # Long-term allocations
        self.add_allocation(
            "long_term_reserve",
            AllocationStage.LONG_TERM,
            AllocationType.RESERVE,
            0.05,  # 5%
            100.0,
            10000.0,
            6
        )
        
        # Organize by stage
        self._organize_allocations_by_stage()
    
    def _organize_allocations_by_stage(self):
        """Organize allocations by stage."""
        self.stage_allocations = {}
        for allocation in self.allocations.values():
            stage = allocation.stage
            if stage not in self.stage_allocations:
                self.stage_allocations[stage] = []
            self.stage_allocations[stage].append(allocation)
        
        # Sort by priority within each stage
        for stage in self.stage_allocations:
            self.stage_allocations[stage].sort(key=lambda x: x.priority)
    
    def add_allocation(self, allocation_id: str, stage: AllocationStage,
                      allocation_type: AllocationType, percentage: float,
                      min_amount: float, max_amount: float, priority: int) -> bool:
        """Add a profit allocation."""
        try:
            allocation = ProfitAllocation(
                allocation_id=allocation_id,
                stage=stage,
                allocation_type=allocation_type,
                percentage=percentage,
                min_amount=min_amount,
                max_amount=max_amount,
                priority=priority
            )
            
            self.allocations[allocation_id] = allocation
            self._organize_allocations_by_stage()
            
            logger.info(f"Added allocation: {allocation_id} ({stage.value}, {percentage:.1%})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding allocation {allocation_id}: {e}")
            return False
    
    def start_profit_cycle(self, cycle_id: Optional[str] = None) -> ProfitCycle:
        """
        Start a new profit allocation cycle.
        
        Args:
            cycle_id: Optional cycle ID, auto-generated if None
            
        Returns:
            New profit cycle
        """
        try:
            if cycle_id is None:
                cycle_id = f"profit_cycle_{int(time.time() * 1000)}"
            
            # End current cycle if exists
            if self.current_cycle:
                self.end_cycle()
            
            # Create new cycle
            self.current_cycle = ProfitCycle(
                cycle_id=cycle_id,
                start_time=time.time()
            )
            
            logger.info(f"Started profit cycle: {cycle_id}")
            return self.current_cycle
            
        except Exception as e:
            logger.error(f"Error starting profit cycle: {e}")
            return self._create_default_cycle()
    
    def _create_default_cycle(self) -> ProfitCycle:
        """Create default profit cycle."""
        return ProfitCycle(
            cycle_id="default_cycle",
            start_time=time.time()
        )
    
    def end_cycle(self) -> Optional[ProfitCycle]:
        """End the current profit cycle."""
        try:
            if not self.current_cycle:
                return None
            
            # Calculate final metrics
            end_time = time.time()
            self.current_cycle.end_time = end_time
            
            # Add to history
            self.cycle_history.append(self.current_cycle)
            if len(self.cycle_history) > self.max_history_size:
                self.cycle_history.pop(0)
            
            # Update metrics
            self.total_cycles += 1
            self.total_profit += self.current_cycle.total_profit
            self.total_allocated += self.current_cycle.allocated_profit
            
            # Reset state
            completed_cycle = self.current_cycle
            self.current_cycle = None
            
            logger.info(f"Ended profit cycle: {completed_cycle.cycle_id} (profit: ${completed_cycle.total_profit:.2f})")
            return completed_cycle
            
        except Exception as e:
            logger.error(f"Error ending profit cycle: {e}")
            return None
    
    def allocate_profit(self, profit_amount: float, 
                       portfolio_state: Dict[str, Any]) -> List[AllocationResult]:
        """
        Allocate profit according to configured strategies.
        
        Args:
            profit_amount: Amount of profit to allocate
            portfolio_state: Current portfolio state
            
        Returns:
            List of allocation results
        """
        try:
            if not self.current_cycle:
                self.start_profit_cycle()
            
            # Update cycle profit
            self.current_cycle.total_profit += profit_amount
            self.current_cycle.remaining_profit += profit_amount
            
            # Check minimum threshold
            if profit_amount < self.config['min_profit_threshold']:
                logger.info(f"Profit ${profit_amount:.2f} below threshold, skipping allocation")
                return []
            
            # Calculate allocations
            allocation_results = []
            remaining_profit = profit_amount
            
            # Process allocations by stage and priority
            for stage in AllocationStage:
                if stage not in self.stage_allocations:
                    continue
                
                stage_allocations = self.stage_allocations[stage]
                stage_results = self._process_stage_allocations(
                    stage_allocations, remaining_profit, portfolio_state
                )
                
                allocation_results.extend(stage_results)
                
                # Update remaining profit
                allocated_in_stage = sum(result.amount for result in stage_results)
                remaining_profit -= allocated_in_stage
            
            # Update cycle
            total_allocated = sum(result.amount for result in allocation_results)
            self.current_cycle.allocated_profit += total_allocated
            self.current_cycle.remaining_profit -= total_allocated
            self.current_cycle.allocations.extend(allocation_results)
            
            # Update metrics
            self.successful_allocations += len([r for r in allocation_results if r.success])
            self.failed_allocations += len([r for r in allocation_results if not r.success])
            
            self.last_update = time.time()
            
            logger.info(f"Allocated ${total_allocated:.2f} of ${profit_amount:.2f} profit")
            return allocation_results
            
        except Exception as e:
            logger.error(f"Error allocating profit: {e}")
            return []
    
    def _process_stage_allocations(self, stage_allocations: List[ProfitAllocation],
                                  available_profit: float,
                                  portfolio_state: Dict[str, Any]) -> List[AllocationResult]:
        """Process allocations for a specific stage."""
        results = []
        remaining_profit = available_profit
        
        for allocation in stage_allocations:
            if not allocation.enabled or remaining_profit <= 0:
                continue
            
            # Calculate allocation amount
            amount = self._calculate_allocation_amount(
                allocation, remaining_profit, portfolio_state
            )
            
            if amount <= 0:
                continue
            
            # Create allocation result
            result = AllocationResult(
                allocation_id=allocation.allocation_id,
                timestamp=time.time(),
                stage=allocation.stage,
                allocation_type=allocation.allocation_type,
                amount=amount,
                percentage=amount / available_profit if available_profit > 0 else 0.0,
                success=True,
                metadata={
                    "allocation_type": allocation.allocation_type.value,
                    "priority": allocation.priority
                }
            )
            
            results.append(result)
            remaining_profit -= amount
            
            logger.debug(f"Allocated ${amount:.2f} to {allocation.allocation_id}")
        
        return results
    
    def _calculate_allocation_amount(self, allocation: ProfitAllocation,
                                   available_profit: float,
                                   portfolio_state: Dict[str, Any]) -> float:
        """Calculate allocation amount based on configuration and constraints."""
        try:
            # Base amount from percentage
            base_amount = available_profit * allocation.percentage
            
            # Apply min/max constraints
            constrained_amount = np.clip(base_amount, allocation.min_amount, allocation.max_amount)
            
            # Apply risk-adjusted allocation if enabled
            if self.config.get('risk_adjusted_allocation', True):
                risk_factor = self._calculate_risk_factor(portfolio_state)
                constrained_amount *= risk_factor
            
            # Ensure we don't exceed available profit
            final_amount = min(constrained_amount, available_profit)
            
            # Apply reserve constraint
            if allocation.allocation_type == AllocationType.RESERVE:
                max_reserve = available_profit * self.config['reserve_percentage']
                final_amount = min(final_amount, max_reserve)
            
            return max(final_amount, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating allocation amount: {e}")
            return 0.0
    
    def _calculate_risk_factor(self, portfolio_state: Dict[str, Any]) -> float:
        """Calculate risk factor for allocation adjustment."""
        try:
            # Simple risk calculation based on portfolio volatility
            volatility = portfolio_state.get('volatility', 0.5)
            total_value = portfolio_state.get('total_value', 1000.0)
            
            # Higher volatility = lower risk factor
            volatility_factor = 1.0 - min(volatility, 1.0)
            
            # Size factor (larger portfolios get more conservative)
            size_factor = 1.0 / (1.0 + total_value / 10000.0)
            
            risk_factor = (volatility_factor + size_factor) / 2
            return np.clip(risk_factor, 0.1, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating risk factor: {e}")
            return 0.5
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of profit allocations."""
        return {
            "total_allocations": len(self.allocations),
            "total_cycles": self.total_cycles,
            "total_profit": self.total_profit,
            "total_allocated": self.total_allocated,
            "allocation_rate": self.total_allocated / self.total_profit if self.total_profit > 0 else 0.0,
            "successful_allocations": self.successful_allocations,
            "failed_allocations": self.failed_allocations,
            "current_cycle": self.current_cycle.cycle_id if self.current_cycle else None,
            "cycle_history_size": len(self.cycle_history),
            "last_update": self.last_update
        }
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """Get summary by allocation stage."""
        stage_summary = {}
        
        for stage in AllocationStage:
            stage_allocations = self.stage_allocations.get(stage, [])
            total_percentage = sum(a.percentage for a in stage_allocations if a.enabled)
            
            stage_summary[stage.value] = {
                "allocations_count": len(stage_allocations),
                "enabled_count": len([a for a in stage_allocations if a.enabled]),
                "total_percentage": total_percentage,
                "allocation_types": list(set(a.allocation_type.value for a in stage_allocations))
            }
        
        return stage_summary
    
    def get_recent_allocations(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent allocation results."""
        if not self.current_cycle:
            return []
        
        recent_allocations = self.current_cycle.allocations[-count:]
        return [
            {
                "allocation_id": result.allocation_id,
                "timestamp": result.timestamp,
                "stage": result.stage.value,
                "allocation_type": result.allocation_type.value,
                "amount": result.amount,
                "percentage": result.percentage,
                "success": result.success
            }
            for result in recent_allocations
        ]
    
    def get_recent_cycles(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent profit cycles."""
        recent_cycles = self.cycle_history[-count:]
        return [
            {
                "cycle_id": cycle.cycle_id,
                "start_time": cycle.start_time,
                "end_time": cycle.end_time,
                "total_profit": cycle.total_profit,
                "allocated_profit": cycle.allocated_profit,
                "remaining_profit": cycle.remaining_profit,
                "allocations_count": len(cycle.allocations)
            }
            for cycle in recent_cycles
        ]
    
    def optimize_allocations(self, performance_data: Dict[str, Any]) -> bool:
        """
        Optimize allocations based on performance data.
        
        Args:
            performance_data: Historical performance data
            
        Returns:
            True if optimization was successful
        """
        try:
            # Simple optimization logic (can be enhanced with ML)
            total_return = performance_data.get('total_return', 0.0)
            volatility = performance_data.get('volatility', 0.5)
            
            # Adjust allocation percentages based on performance
            if total_return > 0.1:  # 10% return
                # Increase reinvestment
                self._adjust_allocation_percentage("immediate_reinvest", 0.05)
                self._adjust_allocation_percentage("short_term_compound", 0.03)
            elif total_return < -0.05:  # -5% return
                # Increase reserve
                self._adjust_allocation_percentage("immediate_reserve", 0.05)
                self._adjust_allocation_percentage("long_term_reserve", 0.03)
            
            # Adjust for volatility
            if volatility > 0.3:
                # High volatility - increase reserve
                self._adjust_allocation_percentage("immediate_reserve", 0.03)
            elif volatility < 0.1:
                # Low volatility - increase reinvestment
                self._adjust_allocation_percentage("immediate_reinvest", 0.03)
            
            logger.info("Optimized allocations based on performance data")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing allocations: {e}")
            return False
    
    def _adjust_allocation_percentage(self, allocation_id: str, adjustment: float):
        """Adjust allocation percentage."""
        if allocation_id in self.allocations:
            allocation = self.allocations[allocation_id]
            new_percentage = np.clip(allocation.percentage + adjustment, 0.0, 1.0)
            allocation.percentage = new_percentage
            logger.debug(f"Adjusted {allocation_id} percentage to {new_percentage:.1%}") 