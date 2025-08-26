from setuptools import setup, find_packages

setup(
    name="bates-rbp",
    version="0.1.0",
    description="An interactive tool for RNA-binding protein (RBP) binding site prediction",
    author="Callum Bates",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "dash",
        "dash-bio",
        "pandas",
        "numpy",
        "requests",
        "biopython",
        "gdown",

    ],
    entry_points={
    "console_scripts": [
        "bates-rbp = bates_rbp.app:main",                  # launches app
        "bates-rbp-download-models = bates_rbp.get_models:main",  # downloads models
    ]
},

)
