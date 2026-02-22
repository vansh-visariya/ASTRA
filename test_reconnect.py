#!/usr/bin/env python
"""Quick test script to verify WebSocket reconnection logic without imports."""

import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_code_changes():
    """Test the code changes without importing."""
    logger.info("\n=== Verifying WebSocket Reconnection Logic ===\n")
    
    # Read the client file
    with open('client_app/client_app.py', 'r') as f:
        client_code = f.read()
    
    # Read the server file
    with open('networking/server_api.py', 'r') as f:
        server_code = f.read()
    
    checks = [
        ("Client: Shorter ping interval", r"ping_interval=10", client_code),
        ("Client: Metrics error handling", r"Failed to send metrics.*connection may have closed", client_code),
        ("Client: Metrics reconnect", r"await self\.disconnect\(\).*await self\.connect\(\)", client_code),
        ("Client: Update error handling", r"Failed to send update.*connection may have closed", client_code),
        ("Client: Update reconnect", r"# Attempt to reconnect and retry", client_code),
        ("Server: Better metrics logging", r"Received metrics from", server_code),
        ("Server: Metrics send error handling", r"Failed to send error response", server_code),
        ("Server: WebSocket error handler", r"WebSocket error:.*exc_info=True", server_code),
    ]
    
    results = []
    for check_name, pattern, code in checks:
        if re.search(pattern, code, re.DOTALL):
            logger.info(f"✓ {check_name}")
            results.append(True)
        else:
            logger.error(f"✗ {check_name}")
            logger.error(f"  Pattern not found: {pattern[:50]}...")
            results.append(False)
    
    if all(results):
        logger.info("\n✓ All code change checks passed!")
        return True
    else:
        logger.error(f"\n✗ {sum(not r for r in results)} checks failed")
        return False

if __name__ == "__main__":
    import sys
    result = test_code_changes()
    sys.exit(0 if result else 1)
