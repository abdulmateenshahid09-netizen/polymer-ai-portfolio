import importlib

packages = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "optuna",
            "matplotlib", "streamlit", "rdkit", "chromadb", "pypdf"]

for p in packages:
    try:
        mod = importlib.import_module(p)
        print(f"OK     {p} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"FAILED {p}: {e}")