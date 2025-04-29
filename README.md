# Detección de Fraudes en Transacciones de Bitcoin

Este proyecto implementa un sistema de detección de fraudes en transacciones de Bitcoin utilizando técnicas de aprendizaje semi-supervisado y análisis de grafos.

## Estructura del Proyecto

```
crypto-aml-risk-detection/
├── config.yaml           # Archivo de configuración
├── data/                 # Directorio para datos
│   └── elliptic_bitcoin_dataset/  # Dataset de Bitcoin
├── models/               # Directorio para modelos entrenados
├── results/              # Directorio para resultados y visualizaciones
├── src/                  # Código fuente
│   ├── data/             # Módulos para carga y preparación de datos
│   ├── features/         # Módulos para extracción de características
│   ├── models/           # Módulos para modelos de aprendizaje
│   ├── visualization/    # Módulos para visualización
│   └── main.py           # Script principal
└── tests/                # Pruebas unitarias
```

## Requisitos

- Python 3.8+
- pandas
- numpy
- scikit-learn
- networkx
- matplotlib
- seaborn
- joblib
- pyyaml

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/crypto-aml-risk-detection.git
cd crypto-aml-risk-detection
```

2. Crear un entorno virtual e instalar dependencias:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Instalar el proyecto en modo desarrollo:
```bash
pip install -e .
```

## Configuración

El proyecto utiliza un archivo de configuración YAML (`config.yaml`) para gestionar parámetros y rutas. Puedes modificar este archivo para ajustar:

- Directorios de datos, resultados y modelos
- Parámetros del modelo semi-supervisado
- Parámetros de extracción de características
- Configuración de visualización
- Opciones de ejecución

Ejemplo de configuración:
```yaml
# Directorios
directories:
  data: "data/elliptic_bitcoin_dataset"
  results: "results"
  models: "models"

# Parámetros del modelo
model:
  semi_supervised:
    n_neighbors: 10
    max_iter: 1000
    n_estimators: 100
    random_state: 42
```

## Uso

### Ejecución Básica

```bash
python src/main.py
```

### Ejecución con Parámetros Personalizados

```bash
python src/main.py --config mi_config.yaml
```

### Ejecución con Argumentos de Línea de Comandos

```bash
python src/main.py --data_dir /ruta/a/datos --results_dir /ruta/a/resultados --n_components 128
```

### Ejecución de Pruebas

```bash
python -m unittest discover tests
```

## Flujo de Trabajo

1. **Carga de Datos**: Se cargan los datos de transacciones de Bitcoin.
2. **Análisis Exploratorio**: Se realiza un análisis exploratorio de los datos (opcional).
3. **Extracción de Características**: Se extraen características de grafo y embeddings.
4. **Preparación de Datos**: Se preparan los datos para el entrenamiento.
5. **Entrenamiento del Modelo Semi-supervisado**: Se entrena un modelo para etiquetar datos no etiquetados.
6. **Entrenamiento del Modelo Final**: Se entrena un modelo final usando todas las etiquetas.
7. **Visualización y Evaluación**: Se generan visualizaciones y se evalúa el rendimiento del modelo.

## Resultados

Los resultados se guardan en el directorio `results/` e incluyen:

- Visualizaciones de distribución de clases
- Visualizaciones de métricas de red
- Visualizaciones de importancia de características
- Visualizaciones de distribución de predicciones
- Métricas de rendimiento del modelo

Los modelos entrenados se guardan en el directorio `models/`.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.