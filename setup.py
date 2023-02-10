from setuptools import setup, find_packages

setup(name='grae',
      version='0.1',
      python_requires='>=3.8',
      description='main package',
      packages=find_packages(),
      install_requires=[
            "numpy<1.22",
            "matplotlib",
            "pandas",
            "pydiffmap",
            "phate",
            "scikit-learn",
            "umap-learn",
            "torch",
            "torchvision",
            "comet-ml",
            "scikit-image",
            "requests",
      ])
