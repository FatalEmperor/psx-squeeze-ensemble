from setuptools import setup, find_packages

setup(
    name="psx-squeeze-ensemble",
    version="1.0.0",
    description="Multi-layered ensemble squeeze strategy for PSX equities",
    author="Haseeb Zahid",
    author_email="haseeb@floretcapitals.com",
    url="https://github.com/haseeb-zahid/psx-squeeze-ensemble",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "requests>=2.28.0",
        "xgboost>=1.7.0",
        "hmmlearn>=0.3.0",
        "scikit-learn>=1.2.0",
        "yfinance>=0.2.0",
    ],
    extras_require={
        "optimize": ["backtesting>=0.3.3", "pandas-ta>=0.3.14b"],
        "dev":      ["pytest>=7.0", "pytest-cov"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
