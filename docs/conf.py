# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Prueba QLIB Jaime'
copyright = '2023, Jaime'
author = 'Jaime'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# JAIME: Cambios metidos por mi segun la pagina con las instrucciones
import os 
import sys
sys.path.insert(0, os.path.abspath('.')) 
sys.path.insert(0, os.path.abspath('..')) 

# extensions = []
extensions = [
    'sphinx.ext.napoleon',      # Supports Google / Numpy docstring 
    'sphinx.ext.autodoc',       # Documentation from docstrings
#    'sphinx.ext.autosummary',    # Autodoc directives
    'sphinx.ext.doctest',       # Test snippets in documentation
    'sphinx.ext.todo',          # to-do syntax highlighting
    'sphinx.ext.ifconfig',      # Content based configuration
    'm2r2',                     # Markdown support 
    'sphinx.ext.viewcode'       # Add links to highlighted source code
]

#autosummary_generate = True

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
#html_theme ='sizzle'  #FALLA!!
#html_theme = 'sphinx_rtd_theme'
#html_theme = 'python_docs_theme'
html_theme = 'bizstyle'

html_static_path = ['_static']

html_theme_options = {
"sidebarwidth": 280,
"globaltoc_includehidden": "True",  
}

# El fichero personalizado debe estar en la carpeta _templates
html_sidebars = { '**': ['jaime_localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'], }

add_module_names = False
toc_object_entries_show_parents = 'hide'
#toc_object_entries = False  #quita demasiadas cosas (NO USAR)

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
#autodoc_class_signature = "separated"