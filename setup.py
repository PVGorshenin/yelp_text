import re
import setuptools

(__version__,) = re.findall("__version__.*\s*=\s*[']([^']+)[']",
                            open('yelp_text/__init__.py').read())

setuptools.setup(
    name="yelp_text",
    version=__version__,
    packages=setuptools.find_packages(),
    python_requires="<3.9.0",
    install_requires=[
        "dill==0.3.3",
        "numpy==1.18.0",
        "pandas==0.25.3",
        "pyyaml==5.4.1",
        "regex==2019.12.20",
        "scikit-learn==0.21.2",
        "scipy==1.4.1",
        "tqdm==4.41.1",
        "transformers==4.0.1",
        "torch==1.7.1",
        "uvicorn==0.13.4",
        "fastapi",
        "gdown"
    ],
)