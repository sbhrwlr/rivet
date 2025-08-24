"""
Callback system for Rivet agents.

Provides event-driven hooks for monitoring and extending agent behavior.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class CallbackEvent(Enum):
    """Standard callback events in agent lifecycle."""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    STREAMING_CHUNK = "streaming_chunk"
    ERROR = "error"


class CallbackManager:
    """
    Manages callbacks for agent lifecycle events.
    
    Supports both sync and async callbacks with error isolation.
    """
    
    def __init__(self):
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._logger = logger.getChild("CallbackManager")
    
    def register(self, event: Union[str, CallbackEvent], callback: Callable) -> None:
        """
        Register a callback for a specific event.
        
        Args:
            event: Event name or CallbackEvent enum
            callback: Function to call when event occurs
        """
        event_name = event.value if isinstance(event, CallbackEvent) else event
        self.callbacks[event_name].append(callback)
        self._logger.debug(f"Registered callback for event: {event_name}")
    
    def unregister(self, event: Union[str, CallbackEvent], callback: Callable) -> bool:
        """
        Unregister a callback for a specific event.
        
        Args:
            event: Event name or CallbackEvent enum
            callback: Function to remove
            
        Returns:
            True if callback was found and removed, False otherwise
        """
        event_name = event.value if isinstance(event, CallbackEvent) else event
        try:
            self.callbacks[event_name].remove(callback)
            self._logger.debug(f"Unregistered callback for event: {event_name}")
            return True
        except ValueError:
            return False
    
    async def trigger(self, event: Union[str, CallbackEvent], data: Any = None) -> None:
        """
        Trigger all callbacks for an event asynchronously.
        
        Args:
            event: Event name or CallbackEvent enum
            data: Data to pass to callbacks
        """
        event_name = event.value if isinstance(event, CallbackEvent) else event
        callbacks = self.callbacks.get(event_name, [])
        
        if not callbacks:
            return
        
        self._logger.debug(f"Triggering {len(callbacks)} callbacks for event: {event_name}")
        
        # Execute all callbacks concurrently with error isolation
        tasks = []
        for callback in callbacks:
            task = asyncio.create_task(self._execute_callback(callback, data, event_name))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def trigger_sync(self, event: Union[str, CallbackEvent], data: Any = None) -> None:
        """
        Trigger all callbacks for an event synchronously.
        
        Args:
            event: Event name or CallbackEvent enum
            data: Data to pass to callbacks
        """
        event_name = event.value if isinstance(event, CallbackEvent) else event
        callbacks = self.callbacks.get(event_name, [])
        
        if not callbacks:
            return
        
        self._logger.debug(f"Triggering {len(callbacks)} callbacks for event: {event_name}")
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # For async callbacks in sync context, we need to run them
                    # This is a fallback - prefer using trigger() for async callbacks
                    self._logger.warning(f"Running async callback {callback.__name__} in sync context")
                    asyncio.run(callback(data))
                else:
                    callback(data)
            except Exception as e:
                self._logger.error(f"Error in callback {callback.__name__} for event {event_name}: {e}")
    
    async def _execute_callback(self, callback: Callable, data: Any, event_name: str) -> None:
        """
        Execute a single callback with error handling.
        
        Args:
            callback: Callback function to execute
            data: Data to pass to callback
            event_name: Name of the event (for logging)
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                # Run sync callback in thread pool to avoid blocking
                await asyncio.to_thread(callback, data)
        except Exception as e:
            self._logger.error(f"Error in callback {callback.__name__} for event {event_name}: {e}")
    
    def clear(self, event: Optional[Union[str, CallbackEvent]] = None) -> None:
        """
        Clear callbacks for a specific event or all events.
        
        Args:
            event: Event to clear callbacks for, or None to clear all
        """
        if event is None:
            self.callbacks.clear()
            self._logger.debug("Cleared all callbacks")
        else:
            event_name = event.value if isinstance(event, CallbackEvent) else event
            self.callbacks[event_name].clear()
            self._logger.debug(f"Cleared callbacks for event: {event_name}")
    
    def get_callbacks(self, event: Union[str, CallbackEvent]) -> List[Callable]:
        """
        Get all callbacks registered for an event.
        
        Args:
            event: Event name or CallbackEvent enum
            
        Returns:
            List of callback functions
        """
        event_name = event.value if isinstance(event, CallbackEvent) else event
        return self.callbacks.get(event_name, []).copy()
    
    def has_callbacks(self, event: Union[str, CallbackEvent]) -> bool:
        """
        Check if any callbacks are registered for an event.
        
        Args:
            event: Event name or CallbackEvent enum
            
        Returns:
            True if callbacks exist for the event
        """
        event_name = event.value if isinstance(event, CallbackEvent) else event
        return len(self.callbacks.get(event_name, [])) > 0