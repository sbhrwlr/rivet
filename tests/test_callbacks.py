"""
Tests for the callback system.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock
from rivet.callbacks import CallbackManager, CallbackEvent


class TestCallbackManager:
    """Test CallbackManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.callback_manager = CallbackManager()
        self.sync_callback = Mock()
        self.async_callback = AsyncMock()
    
    def test_register_callback_with_string_event(self):
        """Test registering callback with string event name."""
        self.callback_manager.register("test_event", self.sync_callback)
        
        callbacks = self.callback_manager.get_callbacks("test_event")
        assert len(callbacks) == 1
        assert callbacks[0] == self.sync_callback
    
    def test_register_callback_with_enum_event(self):
        """Test registering callback with CallbackEvent enum."""
        self.callback_manager.register(CallbackEvent.AGENT_START, self.sync_callback)
        
        callbacks = self.callback_manager.get_callbacks(CallbackEvent.AGENT_START)
        assert len(callbacks) == 1
        assert callbacks[0] == self.sync_callback
    
    def test_register_multiple_callbacks_same_event(self):
        """Test registering multiple callbacks for the same event."""
        callback2 = Mock()
        
        self.callback_manager.register("test_event", self.sync_callback)
        self.callback_manager.register("test_event", callback2)
        
        callbacks = self.callback_manager.get_callbacks("test_event")
        assert len(callbacks) == 2
        assert self.sync_callback in callbacks
        assert callback2 in callbacks
    
    def test_unregister_callback(self):
        """Test unregistering a callback."""
        self.callback_manager.register("test_event", self.sync_callback)
        
        # Verify callback is registered
        assert len(self.callback_manager.get_callbacks("test_event")) == 1
        
        # Unregister and verify
        result = self.callback_manager.unregister("test_event", self.sync_callback)
        assert result is True
        assert len(self.callback_manager.get_callbacks("test_event")) == 0
    
    def test_unregister_nonexistent_callback(self):
        """Test unregistering a callback that doesn't exist."""
        result = self.callback_manager.unregister("test_event", self.sync_callback)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_trigger_sync_callback(self):
        """Test triggering synchronous callback."""
        self.callback_manager.register("test_event", self.sync_callback)
        
        test_data = {"message": "test"}
        await self.callback_manager.trigger("test_event", test_data)
        
        self.sync_callback.assert_called_once_with(test_data)
    
    @pytest.mark.asyncio
    async def test_trigger_async_callback(self):
        """Test triggering asynchronous callback."""
        self.callback_manager.register("test_event", self.async_callback)
        
        test_data = {"message": "test"}
        await self.callback_manager.trigger("test_event", test_data)
        
        self.async_callback.assert_called_once_with(test_data)
    
    @pytest.mark.asyncio
    async def test_trigger_multiple_callbacks(self):
        """Test triggering multiple callbacks for the same event."""
        callback2 = Mock()
        
        self.callback_manager.register("test_event", self.sync_callback)
        self.callback_manager.register("test_event", callback2)
        
        test_data = {"message": "test"}
        await self.callback_manager.trigger("test_event", test_data)
        
        self.sync_callback.assert_called_once_with(test_data)
        callback2.assert_called_once_with(test_data)
    
    @pytest.mark.asyncio
    async def test_trigger_with_callback_error(self):
        """Test that callback errors don't break execution."""
        def error_callback(data):
            raise ValueError("Test error")
        
        self.callback_manager.register("test_event", error_callback)
        self.callback_manager.register("test_event", self.sync_callback)
        
        test_data = {"message": "test"}
        # Should not raise exception
        await self.callback_manager.trigger("test_event", test_data)
        
        # Second callback should still execute
        self.sync_callback.assert_called_once_with(test_data)
    
    @pytest.mark.asyncio
    async def test_trigger_async_callback_error(self):
        """Test that async callback errors don't break execution."""
        async def error_callback(data):
            raise ValueError("Test async error")
        
        self.callback_manager.register("test_event", error_callback)
        self.callback_manager.register("test_event", self.async_callback)
        
        test_data = {"message": "test"}
        # Should not raise exception
        await self.callback_manager.trigger("test_event", test_data)
        
        # Second callback should still execute
        self.async_callback.assert_called_once_with(test_data)
    
    @pytest.mark.asyncio
    async def test_trigger_nonexistent_event(self):
        """Test triggering event with no callbacks."""
        # Should not raise exception
        await self.callback_manager.trigger("nonexistent_event", {"data": "test"})
    
    def test_trigger_sync(self):
        """Test synchronous trigger method."""
        self.callback_manager.register("test_event", self.sync_callback)
        
        test_data = {"message": "test"}
        self.callback_manager.trigger_sync("test_event", test_data)
        
        self.sync_callback.assert_called_once_with(test_data)
    
    def test_trigger_sync_with_error(self):
        """Test synchronous trigger with callback error."""
        def error_callback(data):
            raise ValueError("Test error")
        
        self.callback_manager.register("test_event", error_callback)
        self.callback_manager.register("test_event", self.sync_callback)
        
        test_data = {"message": "test"}
        # Should not raise exception
        self.callback_manager.trigger_sync("test_event", test_data)
        
        # Second callback should still execute
        self.sync_callback.assert_called_once_with(test_data)
    
    def test_clear_specific_event(self):
        """Test clearing callbacks for a specific event."""
        self.callback_manager.register("event1", self.sync_callback)
        self.callback_manager.register("event2", Mock())
        
        self.callback_manager.clear("event1")
        
        assert len(self.callback_manager.get_callbacks("event1")) == 0
        assert len(self.callback_manager.get_callbacks("event2")) == 1
    
    def test_clear_all_events(self):
        """Test clearing all callbacks."""
        self.callback_manager.register("event1", self.sync_callback)
        self.callback_manager.register("event2", Mock())
        
        self.callback_manager.clear()
        
        assert len(self.callback_manager.get_callbacks("event1")) == 0
        assert len(self.callback_manager.get_callbacks("event2")) == 0
    
    def test_has_callbacks(self):
        """Test checking if callbacks exist for an event."""
        assert not self.callback_manager.has_callbacks("test_event")
        
        self.callback_manager.register("test_event", self.sync_callback)
        assert self.callback_manager.has_callbacks("test_event")
        
        self.callback_manager.clear("test_event")
        assert not self.callback_manager.has_callbacks("test_event")
    
    def test_callback_events_enum(self):
        """Test that all standard callback events are defined."""
        expected_events = [
            "agent_start", "agent_end", "model_call", "tool_call", "error"
        ]
        
        for event_name in expected_events:
            # Should be able to find corresponding enum value
            event_enum = getattr(CallbackEvent, event_name.upper())
            assert event_enum.value == event_name
    
    def test_get_callbacks_returns_copy(self):
        """Test that get_callbacks returns a copy, not the original list."""
        self.callback_manager.register("test_event", self.sync_callback)
        
        callbacks = self.callback_manager.get_callbacks("test_event")
        callbacks.append(Mock())  # Modify the returned list
        
        # Original should be unchanged
        original_callbacks = self.callback_manager.get_callbacks("test_event")
        assert len(original_callbacks) == 1
        assert original_callbacks[0] == self.sync_callback


class TestCallbackIntegration:
    """Test callback integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_callback_execution(self):
        """Test that callbacks execute concurrently."""
        callback_manager = CallbackManager()
        
        # Track execution order
        execution_order = []
        
        async def slow_callback(data):
            await asyncio.sleep(0.1)
            execution_order.append("slow")
        
        async def fast_callback(data):
            execution_order.append("fast")
        
        callback_manager.register("test_event", slow_callback)
        callback_manager.register("test_event", fast_callback)
        
        await callback_manager.trigger("test_event", {})
        
        # Fast callback should complete first due to concurrent execution
        assert execution_order == ["fast", "slow"]
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_callbacks(self):
        """Test mixing synchronous and asynchronous callbacks."""
        callback_manager = CallbackManager()
        
        sync_mock = Mock()
        async_mock = AsyncMock()
        
        callback_manager.register("test_event", sync_mock)
        callback_manager.register("test_event", async_mock)
        
        test_data = {"message": "test"}
        await callback_manager.trigger("test_event", test_data)
        
        sync_mock.assert_called_once_with(test_data)
        async_mock.assert_called_once_with(test_data)