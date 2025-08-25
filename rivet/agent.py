"""Core Agent class that orchestrates Models, Memory, and Tools."""

import asyncio
from typing import Any, Dict, List, Optional, AsyncIterator
from .models.base import ModelAdapter
from .memory.base import MemoryAdapter
from .tools import ToolRegistry
from .inspector import Inspector
from .parsers.base import OutputParser
from .callbacks import CallbackManager, CallbackEvent
from .middleware import MiddlewareChain, Middleware
from .streaming import StreamingHandler


class Agent:
    """Central orchestrator for AI agent interactions."""
    
    def __init__(
        self,
        model: ModelAdapter,
        memory: Optional[MemoryAdapter] = None,
        tools: Optional[ToolRegistry] = None,
        inspector: Optional[Inspector] = None,
        output_parser: Optional[OutputParser] = None,
        callbacks: Optional[CallbackManager] = None,
        middleware: Optional[List[Middleware]] = None
    ):
        self.model = model
        self.memory = memory
        self.tools = tools or ToolRegistry()
        self.inspector = inspector or Inspector()
        self.output_parser = output_parser
        self.callbacks = callbacks or CallbackManager()
        self.middleware_chain = MiddlewareChain(middleware or [])
        self.streaming_handler = StreamingHandler(tools=self.tools, callbacks=self.callbacks)
        
    def run(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute agent with a message and optional context."""
        try:
            # Trigger agent start callbacks
            self.callbacks.trigger_sync(CallbackEvent.AGENT_START, {
                "message": message, 
                "context": context
            })
            
            self.inspector.log("agent_start", {"message": message, "context": context})
            
            # Process request through middleware
            request_data = {
                "message": message,
                "context": context or {},
                "_request_id": id(message)
            }
            
            # Run middleware synchronously (fallback for sync context)
            processed_request = asyncio.run(self.middleware_chain.process_request(request_data))
            
            # Extract processed data
            processed_message = processed_request.get("message", message)
            processed_context = processed_request.get("context", context)
            
            # Retrieve relevant memories
            memories = []
            if self.memory:
                memories = self.memory.retrieve(processed_message)
                
            # Build prompt with context and memories
            prompt = self._build_prompt(processed_message, memories, processed_context)
            
            # Trigger model call callback
            self.callbacks.trigger_sync(CallbackEvent.MODEL_CALL, {
                "prompt": prompt,
                "available_tools": self.tools.list()
            })
            
            # Get model response
            response = self.model.generate(prompt, available_tools=self.tools.list())
            
            # Handle tool calls if needed
            if self._has_tool_calls(response):
                self.callbacks.trigger_sync(CallbackEvent.TOOL_CALL, {
                    "response": response,
                    "tools": self.tools.list()
                })
                response = self._execute_tools(response)
                
            # Parse output if parser is configured
            parsed_response = self._parse_output(response)
            
            # Store interaction in memory (store raw response)
            if self.memory:
                self.memory.store(processed_message, response)
            
            # Process response through middleware
            response_data = {
                "result": parsed_response,
                "raw_response": response,
                "_request_id": processed_request.get("_request_id")
            }
            
            processed_response = asyncio.run(self.middleware_chain.process_response(response_data))
            final_result = processed_response.get("result", parsed_response)
            
            self.inspector.log("agent_complete", {"response": response, "parsed_response": final_result})
            
            # Trigger agent end callback
            self.callbacks.trigger_sync(CallbackEvent.AGENT_END, {
                "message": processed_message,
                "response": final_result
            })
            
            return final_result
            
        except Exception as e:
            # Trigger error callback
            self.callbacks.trigger_sync(CallbackEvent.ERROR, {
                "error": str(e),
                "message": message,
                "context": context
            })
            raise
        
    def _build_prompt(self, message: str, memories: List[str], context: Optional[Dict] = None) -> str:
        """Build the complete prompt with memories and context."""
        parts = []
        
        if memories:
            parts.append("Previous context:")
            parts.extend(memories)
            parts.append("")
            
        if context:
            parts.append(f"Additional context: {context}")
            parts.append("")
            
        parts.append(f"User: {message}")
        return "\n".join(parts)
        
    def _has_tool_calls(self, response) -> bool:
        """Check if response contains tool calls."""
        return any(item.type == "function_call" for item in response.output)
        
    def _execute_tools(self, response: str) -> str:
        """Execute any tool calls in the response."""
        # Simple tool execution - can be enhanced
        lines = response.split("\n")
        result_lines = []
        
        for line in lines:
            if "TOOL_CALL:" in line:
                # Extract tool name from anywhere in the line
                tool_call_start = line.find("TOOL_CALL:")
                tool_name = line[tool_call_start + len("TOOL_CALL:"):].strip()
                if tool_name in self.tools.registry:
                    result = self.tools.call(tool_name)
                    result_lines.append(f"TOOL_RESULT: {result}")
                else:
                    result_lines.append(f"TOOL_ERROR: Unknown tool {tool_name}")
            else:
                result_lines.append(line)
                
        return "\n".join(result_lines)
    
    def _parse_output(self, response: str) -> Any:
        """Parse output using configured parser if available."""
        if self.output_parser is None:
            return response
        
        try:
            self.inspector.log("parsing_start", {"parser": type(self.output_parser).__name__})
            parsed = self.output_parser.parse_with_fallback(response)
            self.inspector.log("parsing_success", {"parsed_type": type(parsed).__name__})
            return parsed
        except Exception as e:
            self.inspector.log("parsing_error", {"error": str(e)})
            # Return raw response if parsing fails
            return response
    
    async def arun(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute agent asynchronously with a message and optional context."""
        try:
            # Trigger agent start callbacks
            await self.callbacks.trigger(CallbackEvent.AGENT_START, {
                "message": message, 
                "context": context
            })
            
            self.inspector.log("agent_start", {"message": message, "context": context})
            
            # Process request through middleware
            request_data = {
                "message": message,
                "context": context or {},
                "_request_id": id(message)
            }
            
            processed_request = await self.middleware_chain.process_request(request_data)
            
            # Extract processed data
            processed_message = processed_request.get("message", message)
            processed_context = processed_request.get("context", context)
            
            # Retrieve relevant memories
            memories = []
            if self.memory:
                memories = await self.memory.aretrieve(processed_message)
                
            # Build prompt with context and memories
            prompt = self._build_prompt(processed_message, memories, processed_context)
            
            # Trigger model call callback
            await self.callbacks.trigger(CallbackEvent.MODEL_CALL, {
                "prompt": prompt,
                "available_tools": self.tools.list()
            })
            
            # Get model response
            response = await self.model.agenerate(prompt, tool_registry=self.tools)
            
            # Handle tool calls if needed
            if self._has_tool_calls(response):
                await self.callbacks.trigger(CallbackEvent.TOOL_CALL, {
                    "response": response,
                    "tools": self.tools.list()
                })
                response = await self._execute_tools_async(response)
                
            # Parse output if parser is configured
            parsed_response = self._parse_output(response)
            
            # Store interaction in memory (store raw response)
            if self.memory:
                await self.memory.astore(processed_message, response)
            
            # Process response through middleware
            response_data = {
                "result": parsed_response,
                "raw_response": response,
                "_request_id": processed_request.get("_request_id")
            }
            
            processed_response = await self.middleware_chain.process_response(response_data)
            final_result = processed_response.get("result", parsed_response)
            
            self.inspector.log("agent_complete", {"response": response, "parsed_response": final_result})
            
            # Trigger agent end callback
            await self.callbacks.trigger(CallbackEvent.AGENT_END, {
                "message": processed_message,
                "response": final_result
            })
            
            return final_result
            
        except Exception as e:
            # Trigger error callback
            await self.callbacks.trigger(CallbackEvent.ERROR, {
                "error": str(e),
                "message": message,
                "context": context
            })
            raise
    
    async def astream(self, message: str, context: Optional[Dict[str, Any]] = None) -> AsyncIterator[str]:
        """Stream agent responses asynchronously with tool execution support."""
        try:
            # Trigger agent start callbacks
            await self.callbacks.trigger(CallbackEvent.AGENT_START, {
                "message": message, 
                "context": context,
                "streaming": True
            })
            
            self.inspector.log("agent_stream_start", {"message": message, "context": context})
            
            # Process request through middleware
            request_data = {
                "message": message,
                "context": context or {},
                "_request_id": id(message),
                "streaming": True
            }
            
            processed_request = await self.middleware_chain.process_request(request_data)
            
            # Extract processed data
            processed_message = processed_request.get("message", message)
            processed_context = processed_request.get("context", context)
            
            # Retrieve relevant memories
            memories = []
            if self.memory:
                memories = await self.memory.aretrieve(processed_message)
                
            # Build prompt with context and memories
            prompt = self._build_prompt(processed_message, memories, processed_context)
            
            # Trigger model call callback
            await self.callbacks.trigger(CallbackEvent.MODEL_CALL, {
                "prompt": prompt,
                "available_tools": self.tools.list(),
                "streaming": True
            })
            
            # Get model stream
            model_stream = self.model.stream(prompt, available_tools=self.tools.list())
            
            # Use StreamingHandler to manage tool execution during streaming
            full_response = ""
            stream_context = {
                "message": processed_message,
                "context": processed_context,
                "_request_id": processed_request.get("_request_id")
            }
            
            async for chunk in self.streaming_handler.stream_with_tools(model_stream, stream_context):
                full_response += chunk
                
                # Trigger chunk callback for streaming chunks
                await self.callbacks.trigger(CallbackEvent.STREAMING_CHUNK, {
                    "chunk": chunk,
                    "message": processed_message,
                    "context": stream_context
                })
                
                yield chunk
            
            # Parse final output if parser is configured
            parsed_response = self._parse_output(full_response)
            
            # Store complete interaction in memory (store raw response)
            if self.memory:
                await self.memory.astore(processed_message, full_response)
            
            # Process final response through middleware
            response_data = {
                "result": parsed_response,
                "raw_response": full_response,
                "_request_id": processed_request.get("_request_id"),
                "streaming": True
            }
            
            processed_response = await self.middleware_chain.process_response(response_data)
            
            self.inspector.log("agent_stream_complete", {"response": full_response, "parsed_response": parsed_response})
            
            # Trigger agent end callback with final response
            await self.callbacks.trigger(CallbackEvent.AGENT_END, {
                "message": processed_message,
                "response": processed_response.get("result", parsed_response),
                "raw_response": full_response,
                "streaming": True
            })
            
        except Exception as e:
            # Trigger error callback
            await self.callbacks.trigger(CallbackEvent.ERROR, {
                "error": str(e),
                "message": message,
                "context": context,
                "streaming": True
            })
            raise
    
    async def _execute_tools_async(self, response: str) -> str:
        """Execute tool calls asynchronously."""
        # Use the streaming handler to properly parse and execute tool calls
        return await self.streaming_handler.execute_tool_calls(response)
        
        for line in lines:
            if "TOOL_CALL:" in line:
                # Extract tool name from anywhere in the line
                tool_call_start = line.find("TOOL_CALL:")
                tool_name = line[tool_call_start + len("TOOL_CALL:"):].strip()
                if tool_name in self.tools.registry:
                    # Execute tool using the async call method
                    result = await self.tools.acall(tool_name)
                    result_lines.append(f"TOOL_RESULT: {result}")
                else:
                    result_lines.append(f"TOOL_ERROR: Unknown tool {tool_name}")
            else:
                result_lines.append(line)
        
        return "\n".join(result_lines)