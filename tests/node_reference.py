"""Independent archived upstream entry points, never adapter wrappers."""
import importlib
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'environment/node-reference-site'))
sys.path.insert(0, str(ROOT / 'configs/vendor/node/upstream/qhoptim-wheel'))
package = types.ModuleType('independent_node_reference')
package.__path__ = [str(ROOT / 'configs/vendor/node/upstream/lib')]
sys.modules[package.__name__] = package
nn = importlib.import_module(package.__name__ + '.nn_utils')
arch = importlib.import_module(package.__name__ + '.arch')
from qhoptim.pyt import QHAdam
