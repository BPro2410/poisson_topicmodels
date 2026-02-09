# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add parent directory to path for autodoc
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../packages"))

# -- Project information -------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "poisson-topicmodels"
copyright = "2025, Bernd Prostmaier, Bettina Grün, Paul Hofmarcher"
author = "Bernd Prostmaier, Bettina Grün, Paul Hofmarcher"
release = "0.1.0"
version = "0.1"

# -- General configuration --------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
]

# MyST Parser configuration
myst_parser_config = {
    "enable_math": True,
}

# Autodoc options for better documentation generation
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "both"
add_module_names = False

# Napoleon configuration (Google/NumPy style docstrings)
napoleon_include_init_docstring = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Templates and patterns
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Language and encoding
language = "en"
master_doc = "index"

# -- Options for HTML output -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_logo = "_static/pypf_logo.png"
html_favicon = "_static/pypf_logo.png"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# HTML theme options
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "both",
    "style_external_links": False,
    "vcs_pageview_mode": "view",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# -- Search configuration ---------------------------------------------------
html_search_language = "en"

# -- MathJax configuration --------------------------------------------------
# Use KaTeX for better math rendering (optional)
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    },
}
