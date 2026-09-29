import os
import sys

# Put the repo root on sys.path so `import sim.*` works when pytest is invoked
# from anywhere. tests/ deliberately has no __init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
