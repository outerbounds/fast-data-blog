# Setup

## Python Environment 📦

### (Option A) Use conda
```
mamba env create -f env.yml
```

### (Option B) Use pip
```
python -m venv metaflow-structured-data-env 
source metaflow-structured-data-env/bin/activate
pip install notebook==6.4.10 pyarrow==11.0.0 pandas==1.4.2 matplotlib==3.5.0 duckdb==0.6.0 scipy==1.10.1 lightgbm==3.3.5 seaborn==0.12.1
```

## Run the `FastDataProcessing` flow
```bash
python fast_data_processing.py --environment=conda run
```

## Run the `FastDataModeling` flow
```bash
python fast_data_modeling.py run
```
