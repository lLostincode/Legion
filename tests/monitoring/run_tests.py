import pytest
import os
import sys

# Add the root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

def run_tests():
    """Run all monitoring tests"""
    pytest.main(["-v", os.path.dirname(__file__)])
    
if __name__ == "__main__":
    run_tests() 