# not used since current folder is used as root folder
# will be useful is the whole dir is used as a package
from models.modeling_utils import BaseModel
from models.modeling_utils import BaseConfig

import sys
from pathlib import Path
import importlib
import pkgutil

__version__ = '0.1'

# Import all the models and configs
for _, name, _ in pkgutil.iter_modules([str(Path(__file__).parent / 'models')]):
    print('name:{}'.format(name))
    imported_module = importlib.import_module('.models.' + name, package=__name__)
    for name, cls in imported_module.__dict__.items():
        if isinstance(cls, type) and \
                (issubclass(cls, BaseModel) or issubclass(cls, BaseConfig)):
            setattr(sys.modules[__name__], name, cls)
