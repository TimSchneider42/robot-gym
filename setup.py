from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="robot-gym",
    version="1.0.0",
    description="Provides a modular way of defining learning tasks on the UR10 as gym environments and an abstraction "
                "of simulation and reality to enable seamless switching.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tim Schneider",
    author_email="schneider@ias.informatik.tu-darmstadt.de",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "transformation3d==1.0.1",
        "pyboolet==1.0.1",
        "trimesh==3.7.13",
        "gym>=0.17.2"
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
    ],
)
