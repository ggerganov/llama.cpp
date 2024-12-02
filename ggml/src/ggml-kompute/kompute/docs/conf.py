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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
import sys
import os
import kp
import sphinx_material

project = 'Kompute'
copyright = '2020, The Institute for Ethical AI & Machine Learning'
html_title = "Kompute Documentation (Python & C++)"
author = 'Alejandro Saucedo'

# The full version, including alpha/beta/rc tags
release = '0.8.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    # Creates .nojekyll config
    'sphinx.ext.githubpages',
    # Integrates with doxygen
    "breathe",
    # Converts markdown to rst 
    "m2r2",
]
source_suffix = ['.rst', '.md']

# Breathe conf
breathe_default_project = "Kompute"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Chosen Themes:
# * https://github.com/bashtage/sphinx-material/
# * https://github.com/myyasuda/sphinx_materialdesign_theme
html_theme = 'sphinx_material'

if html_theme == 'sphinx_material':
    html_theme_options = {
        'google_analytics_account': 'G-F9LD9HL8LW',
        'base_url': 'https://kompute.cc',
        'color_primary': 'red',
        'color_accent': 'light-blue',
        'repo_url': 'https://github.com/KomputeProject/kompute/',
        'repo_name': 'Kompute',
        'globaltoc_depth': 2,
        'globaltoc_collapse': False,
        'globaltoc_includehidden': False,
        "repo_type": "github",
        "nav_links": [
            {
                "href": "https://github.com/KomputeProject/kompute/",
                "internal": False,
                "title": "Kompute Repo",
            },
        ]
    }

    extensions.append("sphinx_material")
    html_theme_path = sphinx_material.html_theme_path()
    html_context = sphinx_material.get_html_context()

html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html",  "localtoc.html", "searchbox.html"]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'assets/custom.css',
]


