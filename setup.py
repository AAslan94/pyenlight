from setuptools import setup, find_packages

setup(
    name="enlight_iot",
    version="1.0.0",
    description="A modular framework for Optical and Wireless 6G/IoT scenarios",
    author="Alexandros Aslanidis",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "simpy"
    ],
    python_requires=">=3.8",
)
