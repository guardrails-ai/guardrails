#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev
import sys
from distutils.util import convert_path

from setuptools import Command, find_packages, setup

main_ns = {}
ver_path = convert_path("guardrails/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)


# Package meta-data.
NAME = "guardrails-ai"
DESCRIPTION = "Adding guardrails to large language models."
URL = "https://github.com/shreyar/guardrails"
EMAIL = "shreya.rajpal@gmail.com"
AUTHOR = "Shreya Rajpal"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = main_ns["__version__"]

# What packages are required for this module to be executed?
REQUIRED = [
    "lxml",
    "openai",
    "rich",
    "eliot",
    "eliot-tree",
    "pydantic==1.10.9",
    "typer",
    "griffe",
    "tenacity>=8.1.0",
    "pytest",
]

# Read in docs/requirements.txt
with open("docs/requirements.txt") as f:
    DOCS_REQUIREMENTS = f.read().splitlines()

SQL_REQUIREMENTS = ["sqlvalidator", "sqlalchemy>=2.0.9", "sqlglot"]

SUMMARY_REQUIREMENTS = ["thefuzz", "nltk"]

VECTORDB_REQUIREMENTS = ["faiss-cpu", "numpy", "tiktoken"]

DEV_REQUIREMENTS = [
    "black==22.12.0",
    "isort>=5.12.0",
    "flake8>=3.8.4",
    "docformatter>=1.4",
    "pytest-cov>=2.10.1",
    "pre-commit>=2.9.3",
    "twine",
    "pytest-mock",
    "pypdfium2",
    "pytest",
    "pytest-asyncio",
    *SQL_REQUIREMENTS,
    *VECTORDB_REQUIREMENTS,
] + DOCS_REQUIREMENTS

MANIFEST_REQUIREMENTS = ["manifest-ml"]

PROFANITY_REQUIREMENTS = ["alt-profanity-check"]

CRITIQUE_REQUIREMENTS = ["inspiredco"]

# What packages are optional?
EXTRAS = {
    "dev": DEV_REQUIREMENTS,
    "sql": SQL_REQUIREMENTS,
    "manifest": MANIFEST_REQUIREMENTS,
    "vectordb": VECTORDB_REQUIREMENTS,
    "profanity": PROFANITY_REQUIREMENTS,
    "critique": CRITIQUE_REQUIREMENTS,
    "summary": SUMMARY_REQUIREMENTS,
    "all": [
        *DEV_REQUIREMENTS,
        *MANIFEST_REQUIREMENTS,
        *PROFANITY_REQUIREMENTS,
        *CRITIQUE_REQUIREMENTS,
        *SUMMARY_REQUIREMENTS,
    ],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Build the source and wheel.
        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # Add CLI for "guardrails" command, point to guardrails.cli:cli
    entry_points={
        "console_scripts": ["guardrails=guardrails.cli:cli"],
    },
    install_requires=REQUIRED,
    include_package_data=True,
    extras_require=EXTRAS,
    license="Apache 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)
