"""
Sphinx configuration for GRU Dynamic Beta documentation.
"""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock

# Add package to path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# Mock heavy dependencies that are NOT in docs/requirements.txt.
# Only mock tensorflow/keras (too large for RTD) and optional deps
# that aren't needed for doc generation. Do NOT mock scipy, numpy,
# pandas, scikit-learn, pydantic — those are in docs/requirements.txt.
_MOCK_MODULES = [
    'tensorflow', 'tensorflow.keras', 'tensorflow.keras.layers',
    'tensorflow.keras.optimizers', 'tensorflow.keras.callbacks',
    'tensorflow.keras.models', 'tensorflow.keras.backend',
    'tensorflow.keras.regularizers', 'tensorflow.keras.initializers',
    'tensorflow.keras.utils', 'tensorflow.python',
    'keras', 'keras.layers', 'keras.models',
    'ta', 'ta.momentum', 'ta.trend', 'ta.volatility',
    'yfinance',
    'statsmodels', 'statsmodels.tsa', 'statsmodels.tsa.stattools',
    'statsmodels.stats', 'statsmodels.stats.diagnostic',
    'seaborn',
]
for mod_name in _MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

# -- Project information -----------------------------------------------------

project = 'GRU Dynamic Beta'
copyright = f'{datetime.now().year}, Ahmet Selim Yılmaz'
author = 'Ahmet Selim Yılmaz'

# Version info
release = '0.1.3'
version = '0.1.3'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',           # Auto-generate from docstrings
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx.ext.napoleon',          # Support Google/NumPy docstrings
    'sphinx.ext.viewcode',          # Add links to source code
    'sphinx.ext.intersphinx',       # Link to other projects' docs
    'sphinx.ext.mathjax',           # Math rendering
    'sphinx.ext.githubpages',       # GitHub Pages helper
    'myst_parser',                  # Markdown support
    'nbsphinx',                     # Jupyter notebook support
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autodoc_mock_imports = ['tensorflow', 'keras', 'yfinance', 'ta', 'statsmodels', 'seaborn', 'scipy']
autosummary_generate = True

# Napoleon settings (for Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'tensorflow': ('https://www.tensorflow.org/api_docs/python', None),
}

# Source settings
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_static_path = ['_static']
html_css_files = ['custom.css']

# Custom sidebar
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'searchbox.html',
    ]
}

# HTML settings
html_show_sourcelink = True
html_show_sphinx = False
html_show_copyright = True

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
}

latex_documents = [
    (master_doc, 'grubeta.tex', 'GRU Dynamic Beta Documentation',
     'Ahmet Selim Yılmaz', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'grubeta', 'GRU Dynamic Beta Documentation',
     [author], 1)
]

# -- nbsphinx settings -------------------------------------------------------

nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True

# -- MyST settings -----------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
