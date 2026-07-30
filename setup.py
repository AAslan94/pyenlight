from setuptools import find_packages, setup


setup(
    name="pyenlight",
    version="1.0.0",
    description=(
        "A cross-layer simulation framework for indoor optical "
        "and hybrid optical/RF IoT networks"
    ),
    author="Alexandros Aslanidis",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "simpy",
        "pandas",
    ],
    python_requires=">=3.8",
)
