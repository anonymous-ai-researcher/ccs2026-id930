from setuptools import setup, find_packages

setup(
    name="cpfg-mechanism",
    version="1.0.0",
    description="CPFG: Minimax optimal privacy for similarity search over access-controlled vector databases",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
    extras_require={
        "gpu": ["faiss-gpu>=1.7.4"],
        "embeddings": ["transformers>=4.35", "torch>=2.1"],
    },
)
