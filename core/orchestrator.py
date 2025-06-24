#!/usr/bin/env python3
"""
System Orchestrator - Comprehensive System Coordination and Lifecycle Management
===============================================================================

This module implements the main system orchestrator for Schwabot,
coordinating all components and managing the overall system lifecycle.

Core Functionality:
- System coordination and component management
- Lifecycle management (start, stop, restart)
- Component communication and event routing
- System monitoring and health checks
- Resource allocation and optimization
- Error handling and recovery coordination
"""

import logging
import json
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import queue
import weakref
import traceback

logger = logging.getLogger(__name__)

class OrchestratorState(Enum):
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class ComponentState(Enum):
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"

class EventType(Enum):
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    COMPONENT_START = "component_start"
    COMPONENT_STOP = "component_stop"
    COMPONENT_ERROR = "component_error"
    HEALTH_CHECK = "health_check"
    RESOURCE_ALERT = "resource_alert"
    PERFORMANCE_ALERT = "performance_alert"
    MAINTENANCE_REQUIRED = "maintenance_required"

@dataclass
class ComponentInfo:
    component_id: str
    name: str
    component_type: str
    state: ComponentState
    start_time: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    error_count: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemEvent:
    event_id: str
    event_type: EventType
    timestamp: datetime
    component_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"
    message: str = ""

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_components: int
    total_components: int
    error_rate: float
    response_time: float
    throughput: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ComponentManager:
    """Component lifecycle management."""
    
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.component_instances: Dict[str, Any] = {}
        self.component_dependencies: Dict[str, List[str]] = {}
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        self.health_check_interval = 30  # seconds
        self.max_retries = 3
        self.retry_delay = 5  # seconds
    
    def register_component(self, component_id: str, name: str, component_type: str,
                          instance: Any = None, dependencies: List[str] = None) -> bool:
        """Register a component."""
        try:
            component_info = ComponentInfo(
                component_id=component_id,
                name=name,
                component_type=component_type,
                state=ComponentState.UNKNOWN
            )
            
            self.components[component_id] = component_info
            if instance:
                self.component_instances[component_id] = instance
            if dependencies:
                self.component_dependencies[component_id] = dependencies
            
            logger.info(f"Component registered: {component_id} ({name})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering component {component_id}: {e}")
            return False
    
    def unregister_component(self, component_id: str) -> bool:
        """Unregister a component."""
        try:
            if component_id in self.components:
                del self.components[component_id]
            if component_id in self.component_instances:
                del self.component_instances[component_id]
            if component_id in self.component_dependencies:
                del self.component_dependencies[component_id]
            
            logger.info(f"Component unregistered: {component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering component {component_id}: {e}")
            return False
    
    def start_component(self, component_id: str) -> bool:
        """Start a component."""
        try:
            if component_id not in self.components:
                logger.error(f"Component {component_id} not found")
                return False
            
            component_info = self.components[component_id]
            
            # Check dependencies
            if not self._check_dependencies(component_id):
                logger.error(f"Component {component_id} dependencies not met")
                return False
            
            # Update state
            component_info.state = ComponentState.INITIALIZING
            component_info.start_time = datetime.now()
            
            # Start component instance if available
            if component_id in self.component_instances:
                instance = self.component_instances[component_id]
                if hasattr(instance, 'start'):
                    if not instance.start():
                        component_info.state = ComponentState.ERROR
                        component_info.error_count += 1
                        logger.error(f"Failed to start component {component_id}")
                        return False
            
            component_info.state = ComponentState.ACTIVE
            component_info.last_heartbeat = datetime.now()
            
            logger.info(f"Component started: {component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting component {component_id}: {e}")
            if component_id in self.components:
                self.components[component_id].state = ComponentState.ERROR
                self.components[component_id].error_count += 1
            return False
    
    def stop_component(self, component_id: str) -> bool:
        """Stop a component."""
        try:
            if component_id not in self.components:
                logger.error(f"Component {component_id} not found")
                return False
            
            component_info = self.components[component_id]
            component_info.state = ComponentState.SHUTDOWN
            
            # Stop component instance if available
            if component_id in self.component_instances:
                instance = self.component_instances[component_id]
                if hasattr(instance, 'stop'):
                    try:
                        instance.stop()
                    except Exception as e:
                        logger.error(f"Error stopping component {component_id}: {e}")
            
            logger.info(f"Component stopped: {component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping component {component_id}: {e}")
            return False
    
    def _check_dependencies(self, component_id: str) -> bool:
        """Check if component dependencies are met."""
        try:
            if component_id not in self.component_dependencies:
                return True
            
            dependencies = self.component_dependencies[component_id]
            for dep_id in dependencies:
                if dep_id not in self.components:
                    logger.error(f"Dependency {dep_id} not found for component {component_id}")
                    return False
                
                dep_state = self.components[dep_id].state
                if dep_state != ComponentState.ACTIVE:
                    logger.error(f"Dependency {dep_id} not active (state: {dep_state}) for component {component_id}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking dependencies for {component_id}: {e}")
            return False
    
    def get_component_status(self, component_id: str) -> Optional[ComponentInfo]:
        """Get component status."""
        return self.components.get(component_id)
    
    def get_all_components(self) -> Dict[str, ComponentInfo]:
        """Get all components."""
        return self.components.copy()
    
    def get_active_components(self) -> List[str]:
        """Get list of active component IDs."""
        return [comp_id for comp_id, info in self.components.items() 
                if info.state == ComponentState.ACTIVE]
    
    def update_component_heartbeat(self, component_id: str) -> bool:
        """Update component heartbeat."""
        try:
            if component_id in self.components:
                self.components[component_id].last_heartbeat = datetime.now()
                return True
            return False
        except Exception:
            return False

class EventManager:
    """Event management and routing."""
    
    def __init__(self):
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=10000)
        self.event_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.event_thread = None
    
    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register an event handler."""
        self.event_handlers[event_type].append(handler)
        logger.info(f"Event handler registered for {event_type.value}")
    
    def unregister_handler(self, event_type: EventType, handler: Callable) -> None:
        """Unregister an event handler."""
        if event_type in self.event_handlers:
            if handler in self.event_handlers[event_type]:
                self.event_handlers[event_type].remove(handler)
                logger.info(f"Event handler unregistered for {event_type.value}")
    
    def emit_event(self, event: SystemEvent) -> None:
        """Emit an event."""
        try:
            # Add to history
            self.event_history.append(event)
            
            # Add to queue for processing
            self.event_queue.put(event)
            
            logger.debug(f"Event emitted: {event.event_type.value} - {event.message}")
            
        except Exception as e:
            logger.error(f"Error emitting event: {e}")
    
    def start_event_processing(self) -> bool:
        """Start event processing."""
        try:
            if self.is_running:
                return True
            
            self.is_running = True
            self.event_thread = threading.Thread(target=self._event_processing_loop, daemon=True)
            self.event_thread.start()
            
            logger.info("Event processing started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting event processing: {e}")
            return False
    
    def stop_event_processing(self) -> bool:
        """Stop event processing."""
        try:
            self.is_running = False
            if self.event_thread:
                self.event_thread.join(timeout=5)
            
            logger.info("Event processing stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping event processing: {e}")
            return False
    
    def _event_processing_loop(self) -> None:
        """Event processing loop."""
        while self.is_running:
            try:
                # Get event from queue with timeout
                try:
                    event = self.event_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process event
                self._process_event(event)
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                time.sleep(1)
    
    def _process_event(self, event: SystemEvent) -> None:
        """Process a single event."""
        try:
            handlers = self.event_handlers[event.event_type]
            
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
            
        except Exception as e:
            logger.error(f"Error processing event: {e}")
    
    def get_event_history(self, event_type: Optional[EventType] = None, 
                         limit: int = 100) -> List[SystemEvent]:
        """Get event history."""
        try:
            events = list(self.event_history)
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            return events[-limit:]
            
        except Exception as e:
            logger.error(f"Error getting event history: {e}")
            return []

class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self, component_manager: ComponentManager):
        self.component_manager = component_manager
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: deque = deque(maxlen=1000)
        self.alert_thresholds: Dict[str, float] = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'error_rate': 0.1
        }
        self.is_monitoring = False
        self.monitor_thread = None
    
    def register_health_check(self, check_name: str, check_function: Callable) -> None:
        """Register a health check."""
        self.health_checks[check_name] = check_function
        logger.info(f"Health check registered: {check_name}")
    
    def start_monitoring(self) -> bool:
        """Start health monitoring."""
        try:
            if self.is_monitoring:
                return True
            
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            logger.info("Health monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting health monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop health monitoring."""
        try:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            
            logger.info("Health monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping health monitoring: {e}")
            return False
    
    def _monitoring_loop(self) -> None:
        """Health monitoring loop."""
        while self.is_monitoring:
            try:
                # Run health checks
                health_status = self._run_health_checks()
                
                # Store health status
                self.health_history.append(health_status)
                
                # Check for alerts
                self._check_alerts(health_status)
                
                # Sleep between checks
                time.sleep(30)  # 30 second interval
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(30)
    
    def _run_health_checks(self) -> SystemMetrics:
        """Run all health checks."""
        try:
            import psutil
            
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            network = psutil.net_io_counters()
            
            # Component metrics
            components = self.component_manager.get_all_components()
            active_components = len([c for c in components.values() if c.state == ComponentState.ACTIVE])
            total_components = len(components)
            
            # Calculate error rate
            error_count = sum(c.error_count for c in components.values())
            total_operations = max(total_components, 1)
            error_rate = error_count / total_operations
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                },
                active_components=active_components,
                total_components=total_components,
                error_rate=error_rate,
                response_time=0.0,  # Would be calculated from actual measurements
                throughput=0.0  # Would be calculated from actual measurements
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running health checks: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                active_components=0,
                total_components=0,
                error_rate=1.0,
                response_time=0.0,
                throughput=0.0
            )
    
    def _check_alerts(self, metrics: SystemMetrics) -> None:
        """Check for alerts based on metrics."""
        try:
            alerts = []
            
            if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
                alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            
            if metrics.memory_usage > self.alert_thresholds['memory_usage']:
                alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
            
            if metrics.disk_usage > self.alert_thresholds['disk_usage']:
                alerts.append(f"High disk usage: {metrics.disk_usage:.1f}%")
            
            if metrics.error_rate > self.alert_thresholds['error_rate']:
                alerts.append(f"High error rate: {metrics.error_rate:.2f}")
            
            # Emit alerts
            for alert in alerts:
                event = SystemEvent(
                    event_id=f"alert_{int(time.time())}",
                    event_type=EventType.RESOURCE_ALERT,
                    timestamp=datetime.now(),
                    severity="warning",
                    message=alert
                )
                # This would be sent to the event manager
                logger.warning(f"Health alert: {alert}")
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        try:
            if not self.health_history:
                return {'status': 'no_data'}
            
            latest_metrics = self.health_history[-1]
            
            return {
                'timestamp': latest_metrics.timestamp.isoformat(),
                'cpu_usage': latest_metrics.cpu_usage,
                'memory_usage': latest_metrics.memory_usage,
                'disk_usage': latest_metrics.disk_usage,
                'active_components': latest_metrics.active_components,
                'total_components': latest_metrics.total_components,
                'error_rate': latest_metrics.error_rate,
                'status': 'healthy' if latest_metrics.error_rate < 0.1 else 'degraded'
            }
            
        except Exception as e:
            logger.error(f"Error getting health summary: {e}")
            return {'status': 'error'}

class SystemOrchestrator:
    """Main system orchestrator."""
    
    def __init__(self):
        self.state = OrchestratorState.INITIALIZING
        self.component_manager = ComponentManager()
        self.event_manager = EventManager()
        self.health_monitor = HealthMonitor(self.component_manager)
        self.start_time = None
        self.config: Dict[str, Any] = {}
        
        # Register default event handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default event handlers."""
        self.event_manager.register_handler(EventType.COMPONENT_ERROR, self._handle_component_error)
        self.event_manager.register_handler(EventType.RESOURCE_ALERT, self._handle_resource_alert)
        self.event_manager.register_handler(EventType.MAINTENANCE_REQUIRED, self._handle_maintenance_required)
    
    def _handle_component_error(self, event: SystemEvent) -> None:
        """Handle component error events."""
        logger.error(f"Component error: {event.message}")
        # Could implement automatic recovery here
    
    def _handle_resource_alert(self, event: SystemEvent) -> None:
        """Handle resource alert events."""
        logger.warning(f"Resource alert: {event.message}")
        # Could implement resource management here
    
    def _handle_maintenance_required(self, event: SystemEvent) -> None:
        """Handle maintenance required events."""
        logger.info(f"Maintenance required: {event.message}")
        # Could implement maintenance scheduling here
    
    async def start(self) -> bool:
        """Start the orchestrator."""
        try:
            self.state = OrchestratorState.STARTING
            self.start_time = datetime.now()
            
            logger.info("Starting system orchestrator")
            
            # Start event processing
            if not self.event_manager.start_event_processing():
                logger.error("Failed to start event processing")
                return False
            
            # Start health monitoring
            if not self.health_monitor.start_monitoring():
                logger.error("Failed to start health monitoring")
                return False
            
            # Start all components
            if not await self._start_all_components():
                logger.error("Failed to start all components")
                return False
            
            self.state = OrchestratorState.RUNNING
            logger.info("System orchestrator started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting orchestrator: {e}")
            self.state = OrchestratorState.ERROR
            return False
    
    async def stop(self) -> bool:
        """Stop the orchestrator."""
        try:
            self.state = OrchestratorState.STOPPING
            logger.info("Stopping system orchestrator")
            
            # Stop all components
            if not await self._stop_all_components():
                logger.error("Failed to stop all components")
            
            # Stop health monitoring
            self.health_monitor.stop_monitoring()
            
            # Stop event processing
            self.event_manager.stop_event_processing()
            
            self.state = OrchestratorState.STOPPED
            logger.info("System orchestrator stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping orchestrator: {e}")
            self.state = OrchestratorState.ERROR
            return False
    
    async def restart(self) -> bool:
        """Restart the orchestrator."""
        try:
            logger.info("Restarting system orchestrator")
            
            if not await self.stop():
                return False
            
            await asyncio.sleep(2)  # Brief pause
            
            if not await self.start():
                return False
            
            logger.info("System orchestrator restarted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error restarting orchestrator: {e}")
            return False
    
    async def _start_all_components(self) -> bool:
        """Start all registered components."""
        try:
            components = self.component_manager.get_all_components()
            
            for component_id in components:
                if not self.component_manager.start_component(component_id):
                    logger.error(f"Failed to start component {component_id}")
                    return False
            
            logger.info(f"All {len(components)} components started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting components: {e}")
            return False
    
    async def _stop_all_components(self) -> bool:
        """Stop all registered components."""
        try:
            components = self.component_manager.get_all_components()
            
            for component_id in components:
                if not self.component_manager.stop_component(component_id):
                    logger.error(f"Failed to stop component {component_id}")
            
            logger.info(f"All {len(components)} components stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping components: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        try:
            components = self.component_manager.get_all_components()
            active_components = self.component_manager.get_active_components()
            health_summary = self.health_monitor.get_health_summary()
            
            return {
                'state': self.state.value,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'total_components': len(components),
                'active_components': len(active_components),
                'component_states': {
                    comp_id: info.state.value for comp_id, info in components.items()
                },
                'health': health_summary,
                'event_count': len(self.event_manager.event_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'state': self.state.value, 'error': str(e)}
    
    def register_component(self, component_id: str, name: str, component_type: str,
                          instance: Any = None, dependencies: List[str] = None) -> bool:
        """Register a component with the orchestrator."""
        return self.component_manager.register_component(
            component_id, name, component_type, instance, dependencies
        )
    
    def emit_event(self, event: SystemEvent) -> None:
        """Emit an event through the orchestrator."""
        self.event_manager.emit_event(event)

def main():
    """Main function for testing."""
    try:
        # Create orchestrator
        orchestrator = SystemOrchestrator()
        
        # Register some test components
        orchestrator.register_component("test_component_1", "Test Component 1", "test")
        orchestrator.register_component("test_component_2", "Test Component 2", "test")
        
        # Start orchestrator
        import asyncio
        success = asyncio.run(orchestrator.start())
        
        if success:
            print("Orchestrator started successfully!")
            
            # Get status
            status = orchestrator.get_status()
            print("Orchestrator Status:")
            print(json.dumps(status, indent=2, default=str))
            
            # Stop orchestrator
            asyncio.run(orchestrator.stop())
            print("Orchestrator stopped")
        else:
            print("Orchestrator failed to start!")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 