Welcome to the InfraRAPS documentation!
========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Introduction
------------
InfraRAPS (Infrastructure Resilience And Preparedness Simulator) is a python framework modeling coupled infrastructure networks and their resilience to extreme events.
It simulates the US energy and natural gas grid as an optimized coupled network combined with a diffusion-based perturbation network to see how the network performance evolves under critical conditions.
In the current version the perturbation models a military attack with nuclear weapons.

Requirements
------------
Make sure you have all required packages installed:

`pip install -r requirements.txt`

Results
-------
The results were visualized with Dash and can be found online under:

https://manuelk.pythonanywhere.com/

Local display: If you want to display the Dash app locally, run the Dash app via:

`python3 final-project/app.py`

Results are then visible with a browser at http://127.0.0.1:8050/.

Modules
-------

.. automodule:: final_project.optimization
    :members:

.. automodule:: final_project.perturb
    :members:

.. automodule:: final_project.model
    :members:

.. automodule:: final_project.protocols
    :members:

.. automodule:: final_project.app
    :members: