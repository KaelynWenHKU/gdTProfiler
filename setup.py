from setuptools import setup, find_packages

with open("README.rst", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gdTProfiler",
    version="1.0.0",
    description="Identification, Vδ1/Vδ2 subtyping and cell-state profiling of human γδ T cells from single-cell transcriptomes",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/KaelynWenHKU/gdTProfiler",
    author="Ka Ning Wen, Yuanhua Huang, Sung Hee Park",
    license="Apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=find_packages(exclude=["tests"]),
    package_data={"gdtprofiler": ["data/softmax_v1/*", "data/programmes_K8/*", "data/example/*"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=["numpy>=1.23", "pandas>=1.5", "scipy>=1.9", "scikit-learn>=1.2", "anndata>=0.8"],
    extras_require={"plot": ["matplotlib>=3.6"], "test": ["pytest>=7"]},
)
