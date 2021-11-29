# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# path to the module code for autodoc
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../pymice"))


# -- Project information -----------------------------------------------------

project = "pymice"
copyright = "2021, Rianne Schouten & Davina Zamanzadeh"
author = "Rianne Schouten & Davina Zamanzadeh"

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#getting-started
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.autosummary"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# Theme to match pandas
# https://github.com/pandas-dev/pandas/blob/master/doc/source/conf.py#L218
# https://sphinx-themes.org/sample-sites/pydata-sphinx-theme/
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
