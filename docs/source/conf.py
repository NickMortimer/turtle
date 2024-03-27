# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'turtle-drone'
copyright = '2024, Nick Mortimer'
author = 'Nick Mortimer'
release = 'alpha'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinxcontrib.jquery',  # to fix read the docs jQuery bug (https://github.com/readthedocs/sphinx_rtd_theme/issues/1452)
]

templates_path = ['_templates']
exclude_patterns = ['**.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
#html_static_path = ['_static']
