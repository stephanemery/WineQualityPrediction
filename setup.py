from setuptools import setup, find_packages


def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]


setup(
    name="wineQualityPred",
    version="1.0.7",
    description="Wine quality prediction from its physicochimical properties",
    url="https://github.com/stephanemery/WineQualityPrediction",
    license="MIT",
    author="Stéphane Emery / Nassim Augsburger",
    author_email="john@doe.ch",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    entry_points={"console_scripts": ["wineQualityPred = wineQualityPred.paper:reproduceResults"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
