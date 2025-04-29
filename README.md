# Bitcoin Transaction Fraud Detection

This project implements a fraud detection system for Bitcoin transactions using semi-supervised learning techniques and graph analysis.

## Project Structure

```
crypto-aml-risk-detection/
├── config/               # Configuration directory
│   └── config.yaml       # Configuration file
├── data/                 # Data directory
│   └── elliptic_bitcoin_dataset/  # Bitcoin dataset
├── models/               # Directory for trained models
├── results/              # Directory for results and visualizations
├── src/                  # Source code
│   ├── data/             # Modules for data loading and preparation
│   ├── features/         # Modules for feature extraction
│   ├── models/           # Modules for learning models
│   ├── visualization/    # Modules for visualization
│   └── main.py           # Main script
└── tests/                # Unit tests
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- networkx
- matplotlib
- seaborn
- joblib
- pyyaml

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/crypto-aml-risk-detection.git
cd crypto-aml-risk-detection
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install the project in development mode:
```bash
pip install -e .
```

## Configuration

The project uses a YAML configuration file (`config/config.yaml`) to manage parameters and paths. You can modify this file to adjust:

- Data, results, and model directories
- Semi-supervised model parameters
- Feature extraction parameters
- Visualization settings
- Execution options

Example configuration:
```yaml
# Directories
directories:
  data: "data/elliptic_bitcoin_dataset"
  results: "results"
  models: "models"

# Model parameters
model:
  semi_supervised:
    n_neighbors: 10
    max_iter: 1000
    n_estimators: 100
    random_state: 42
```

## Usage

### Basic Execution

```bash
python src/main.py
```

### Execution with Custom Parameters

```bash
python src/main.py --config config/my_config.yaml
```

### Execution with Command Line Arguments

```bash
python src/main.py --data_dir /path/to/data --results_dir /path/to/results --n_components 128
```

### Execution with Exploratory Data Analysis

```bash
python src/main.py --run_eda
```

### Execution without Exploratory Data Analysis

```bash
python src/main.py --skip_eda
```

## Features

- **Semi-supervised Learning**: Utilizes both labeled and unlabeled data to improve fraud detection
- **Graph Analysis**: Extracts features from the Bitcoin transaction network
- **Feature Engineering**: Creates comprehensive feature sets from transaction data
- **Visualization**: Generates informative visualizations of results and model performance
- **Model Evaluation**: Provides detailed metrics and performance analysis

## Results

The system generates the following outputs:

- Trained models saved in the `models/` directory
- Visualizations and analysis results in the `results/` directory
- Performance metrics and classification reports

## Testing

Run the unit tests with:

```bash
python -m unittest discover tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Elliptic Bitcoin dataset for providing transaction data
- Contributors and maintainers of the libraries used in this project