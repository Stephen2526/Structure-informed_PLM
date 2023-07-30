'''
from models.modeling_utils import BaseModel
from models.modeling_utils import BaseConfig

import sys
from pathlib import Path
import importlib
import pkgutil

print('__file__:{}'.format(Path(__file__).parent))

# Import all the models and configs
for _, name, _ in pkgutil.iter_modules([str(Path(__file__).parent)]):
    print('file:{}, folder:{}'.format(name,__name__))  
    imported_module = importlib.import_module('models.' + name, package=__name__)
    for name, cls in imported_module.__dict__.items():
        if isinstance(cls, type) and \
              (issubclass(cls, BaseModel) or issubclass(cls, BaseConfig)):
            setattr(sys.modules[__name__], name, cls)
'''