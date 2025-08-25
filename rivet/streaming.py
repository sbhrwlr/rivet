"""Streaming handler for managing streaming responses with tool call detection."""

import re
import asyncio
from typing import AsyncIterator, List, Dict, Any, Optional, Tuple, Union
from .tools import ToolRegistry
from .callbacks import CallbackManager, CallbackEvent
from .exceptions import StreamingError


class StreamingHandler:
    """Handles streaming responses with tool call detection and execution."""
    
    def __init__(
        self,
        tools: Optional[ToolRegistry] = None,
        callbacks: Optional[CallbackManager] = None
    ):
        self.tools = tools or ToolRegistry()
        self.callbacks = callbacks or CallbackManager()
        self._buffer = ""
        self._tool_call_pattern = re.compile(r'TOOL_CALL:\s*(\w+)(?:\s*\((.*?)\))?')
        
    async def stream_with_tools(
        self,
        stream: AsyncIterator[str],
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """Stream response chunks while detecting and executing tool calls."""
        try:
            full_response = ""
            pending_tool_calls = []
            
            async for chunk in stream:
                full_response += chunk
                self._buffer += chunk
                
                # Check for complete tool calls in buffer
                tool_calls = self._extract_tool_calls(self._buffer)
                
                if tool_calls:
                    # Execute tool calls
                    for tool_name, args in tool_calls:
                        await self.callbacks.trigger(CallbackEvent.TOOL_CALL, {
                            "tool_name": tool_name,
                            "args": args,
                            "context": context,
                            "streaming": True
                        })
                        
                        try:
                            if args:
                                # Parse arguments if provided
                                parsed_args = self._parse_tool_args(args)
                                result = await self.tools.acall(tool_name, **parsed_args)
                            else:
                                result = await self.tools.acall(tool_name)
                            
                            tool_result = f"TOOL_RESULT: {result}"
                            yield tool_result
                            
                        except Exception as e:
                            error_result = f"TOOL_ERROR: {str(e)}"
                            yield error_result
                    
                    # Clear processed tool calls from buffer
                    self._buffer = self._remove_processed_tool_calls(self._buffer, tool_calls)
                else:
                    # No tool calls, yield the chunk
                    yield chunk
                    
        except Exception as e:
            raise StreamingError(f"Streaming with tool execution failed: {str(e)}") from e
        finally:
            # Clear buffer
            self._buffer = ""
    
    def _extract_tool_calls(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """Extract complete tool calls from text buffer."""
        tool_calls = []
        matches = self._tool_call_pattern.findall(text)
        
        for match in matches:
            tool_name = match[0]
            args = match[1] if match[1] else None
            
            # Only add tool calls that have a valid tool name in the registry
            if tool_name in self.tools.registry:
                tool_calls.append((tool_name, args))
            
        return tool_calls
    
    def _parse_tool_args(self, args_str: str) -> Dict[str, Any]:
        """Parse tool arguments from string format."""
        # Simple argument parsing - can be enhanced for more complex formats
        args = {}
        if not args_str.strip():
            return args
            
        # Handle simple key=value pairs
        try:
            # Try to evaluate as Python dict/kwargs
            if args_str.startswith('{') and args_str.endswith('}'):
                # JSON-like format
                import json
                args = json.loads(args_str)
            else:
                # Simple key=value format
                pairs = args_str.split(',')
                has_valid_pairs = False
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        
                        # Try to convert to appropriate type
                        try:
                            # Try int first
                            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                                args[key] = int(value)
                            # Try float
                            elif '.' in value and value.replace('.', '').replace('-', '').isdigit():
                                args[key] = float(value)
                            # Try boolean
                            elif value.lower() in ('true', 'false'):
                                args[key] = value.lower() == 'true'
                            else:
                                args[key] = value
                            has_valid_pairs = True
                        except ValueError:
                            args[key] = value
                            has_valid_pairs = True
                
                # If no valid key=value pairs found, treat as raw input
                if not has_valid_pairs:
                    args = {"input": args_str}
        except Exception:
            # If parsing fails, pass the raw string as a single argument
            args = {"input": args_str}
            
        return args
    
    def _remove_processed_tool_calls(self, text: str, tool_calls: List[Tuple[str, Optional[str]]]) -> str:
        """Remove processed tool calls from buffer."""
        for tool_name, args in tool_calls:
            if args:
                pattern = f"TOOL_CALL:\\s*{re.escape(tool_name)}\\s*\\({re.escape(args)}\\)"
            else:
                pattern = f"TOOL_CALL:\\s*{re.escape(tool_name)}"
            text = re.sub(pattern, "", text, count=1)
        return text
    
    def detect_tool_calls(self, text: str) -> List[str]:
        """Detect tool call names in text."""
        matches = self._tool_call_pattern.findall(text)
        return [match[0] for match in matches]
    
    def has_tool_calls(self, text: str) -> bool:
        """Check if text contains any tool calls."""
        return bool(self._tool_call_pattern.search(text))
    
    async def execute_tool_calls(
        self,
        response: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute all tool calls from a model response and replace them with their results.

        Args:
            response: Model response object containing `tools` and `text`.
            context: Optional context to pass to callbacks.

        Returns:
            str: The response text with tool calls replaced by their results.

        Raises:
            ToolExecutionError: If tool execution or replacement fails.
        """
        if not hasattr(response, "tools") or not hasattr(response, "text"):
            raise ValueError("Response must have `tools` and `text` attributes")

        result_text: str = response.text
        tool_calls: List[Any] = getattr(response, "tools", [])
        
        if not tool_calls:
            return result_text

        for tool_call in tool_calls:
            tool_name: str = getattr(tool_call, "name", "")
            args: Union[str, Dict[str, Any], None] = getattr(response.output, "arguments", None)

            if not tool_name:
                continue

            try:
                # Trigger callback before execution
                await self.callbacks.trigger(
                    CallbackEvent.TOOL_CALL,
                    {
                        "tool_name": tool_name,
                        "args": args,
                        "context": context or {},
                    },
                )

                # Parse and call tool
                if args:
                    parsed_args = self._parse_tool_args(args)
                    result = await self.tools.acall(tool_name, **parsed_args)
                else:
                    result = await self.tools.acall(tool_name)

                replacement = f"TOOL_RESULT: {result}"

            except Exception as e:
                replacement = f"TOOL_ERROR: {str(e)}"

            # Replace in text
            call_repr = self._format_tool_call(tool_name, args)
            if call_repr in result_text:
                result_text = result_text.replace(call_repr, replacement, 1)

        return result_text