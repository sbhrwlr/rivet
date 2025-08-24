#!/usr/bin/env python3
"""
Verification script to ensure the Rivet project is properly set up.
"""

import sys
import rivet
from rivet import Agent

def main():
    print("🔧 Rivet Setup Verification")
    print("=" * 40)
    
    # Check Python version
    print(f"✅ Python version: {sys.version}")
    
    # Check Rivet import
    print(f"✅ Rivet version: {rivet.__version__ if hasattr(rivet, '__version__') else 'imported successfully'}")
    
    # Test basic agent creation
    try:
        agent = Agent(model="mock")
        print("✅ Agent creation: Success")
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False
    
    # Test tool registration
    try:
        @rivet.tool
        def test_tool(message: str) -> str:
            """A simple test tool."""
            return f"Echo: {message}"
        
        print("✅ Tool registration: Success")
    except Exception as e:
        print(f"❌ Tool registration failed: {e}")
        return False
    
    print("\n🎉 Setup verification complete! Your Rivet environment is ready.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)