from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    # lowercase name so that:
    # $ pip install evoxbench
    # instead of
    # $ pip install EvoXBench
    name="evoxbench",
    version="1.0.2",
    description="A benchmark for NAS algorithms",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",
    url="https://github.com/EMI-Group/evoxbench",  # Optional
    author="EMI-Group",  # Optional
    author_email="emi-group@outlook.com",  # Optional
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a list of additional keywords, separated
    # by commas, to be used to assist searching for the distribution in a
    # larger catalog.
    keywords="benchmark, evolution algorithm, neural architecture search",  # Optional
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    # package_dir={"": ""},
    # find_packages failed to work, so manually list packages.
    # data_files=[('', ['evoxbench/__main__.py'])],
    packages=find_packages(),
    # packages=[
    #     "evoxbench",
    #     "evoxbench.benchmarks", "evoxbench.api", "evoxbench.modules",
    #     "evoxbench.api.pymoo"
    #     ],  # Required
    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires=">=3.7, <4",
    # Only top-level dependencies
    install_requires=[
        "pyyaml",
        "django",
        "numpy",
        # "pymoo",
        "scikit-learn",
        "scipy",
    ],  # Optional
    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    # extras_require={  # Optional
    #     "dev": ["check-manifest"],
    # },

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `evoxbenchrpc` which
    # executes the function `main` from this package when invoked:
    entry_points={
        "console_scripts": [
            # "evoxbench=evoxbench.__main__:main",
            "evoxbenchrpc=evoxbench.api.rpc:main",
        ],
    },
    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        "Bug Reports": "https://github.com/EMI-Group/evoxbench/issues",
        "Source": "https://github.com/EMI-Group/evoxbench",
    },

    zip_safe=False,
)
