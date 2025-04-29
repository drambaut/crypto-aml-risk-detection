import os
import sys

# Agregar el directorio src al path para poder importar los módulos
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path) 