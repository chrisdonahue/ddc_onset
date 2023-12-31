from setuptools import setup

setup(
    name="ddc_onset",
    version="0.1",
    description="Onset detector from Dance Dance Convolution in PyTorch",
    url="https://github.com/chrisdonahue/ddc_onset",
    author="Chris Donahue",
    packages=["ddc_onset"],
    package_data={"ddc_onset": ["weights/*"]},
    include_package_data=True,
    install_requires=[
        "numpy",
        "torch>=1.9.0",
    ],
)
