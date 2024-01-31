# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'quantificationlib'
copyright = '2024, AIC Gijón'
author = 'Alberto Castaño, Pablo González, Jaime Alonso, Pablo Pérez, Juan José del Coz'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os 
import sys
sys.path.insert(0, os.path.abspath('.')) 
sys.path.insert(0, os.path.abspath('..')) 

# extensions = []
extensions = [
    'sphinx.ext.napoleon',      # Supports Google / Numpy docstring 
    'sphinx.ext.autodoc',       # Documentation from docstrings
    'sphinx.ext.doctest',       # Test snippets in documentation
    'sphinx.ext.todo',          # to-do syntax highlighting
    'sphinx.ext.ifconfig',      # Content based configuration
    'm2r2',                     # Markdown support 
    'sphinx.ext.viewcode',      # Add links to highlighted source code
]

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'bizstyle'
# html_static_path = ['_static']

html_theme_options = {
"sidebarwidth": 280,
}

html_show_sourcelink = False


# Customlocaltoc must be in _templates
html_sidebars = { '**': ['custom_localtoc.html', 'mylink.html', 'relations.html', 'sourcelink.html', 'searchbox.html'], }

add_module_names = False
toc_object_entries_show_parents = 'hide'


# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
#autodoc_class_signature = "separated"