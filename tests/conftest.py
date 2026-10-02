import os
import sys

# Put the repo root on sys.path so `import simulator.*` and `import maara.*`
# work when pytest is invoked from anywhere. tests/ and its subdirectories
# deliberately have no __init__.py, so this conftest applies to all of them.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
