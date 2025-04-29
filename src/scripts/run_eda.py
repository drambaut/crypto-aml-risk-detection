#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ejecutar el análisis exploratorio de datos.
"""

import os
import sys

# Agregar el directorio src al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar el script de análisis exploratorio
from scripts.exploratory_data_analysis import *

if __name__ == "__main__":
    print("Ejecutando análisis exploratorio de datos...")
    # El script exploratory_data_analysis.py ya contiene el código para ejecutar el análisis
    # y se ejecutará automáticamente al importarlo
    print("Análisis exploratorio completado.") 