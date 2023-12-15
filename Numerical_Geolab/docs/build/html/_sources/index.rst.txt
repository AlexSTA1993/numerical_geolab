.. Numerical Geolab documentation master file, created by
   sphinx-quickstart on Fri Aug 24 17:40:35 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
  
Numerical Geolab's documentation
================================
.. image:: _images/CoQuake_Banner2.png
   :alt: ERC-CoQuake project logo
   :target: http://www.coquake.com
   :align: center

About
-----

*Numerical Geolab* codes and algorithms serve as a basic ingredient for the numerical developments of the *CoQuake* project.

CoQuake project receives funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 757848 "Controlling earthQuakes").

For more details visit: :xref:`CoQuake Project`

Contact: :xref:`Ioannis Stefanou`

The Finite Element module builts on : :xref:`FEniCS project` (open-source, under :xref:`LGPLv3`). 

.. todo::
   * Machine Learning (in progress...)
   * Discrete Elements
   * Large displacements/deformations (at the moment use of the ALE module available in FEniCS)
   * Contact/Interfaces
   * Improve accuracy of diffusion in unit-tests (use of a centered finite difference algorithm for the time discretization)
   
Project structure
-----------------
.. image:: _images/classes.png
   :alt: Structure of classes
   :target: _images/classes.png
   :align: center

Project modules
...............

.. toctree::
   :numbered:
   :maxdepth: 2

   modules


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Numerical Geolab ReadMe
-----------------------
.. toctree::
   :numbered:
   :maxdepth: 2

   installation_instructions
   material_description_state_variables
   Solver_Flowchart
   Definition_of Boundary_Conditions
   
Numerical Geolab Theory
-----------------------
.. toctree::
   :numbered:
   :maxdepth: 2
   
   Linear_and_Bilinear_forms_in_Numerical Geolab

.. _Numerical_Geolab_Tutorials2:   

Numerical Geolab Tutorials
--------------------------

.. toctree::
   :numbered:
   :maxdepth: 1

   Tutorial_Cauchy_elastoplasticity_VM
   Tutorial_Cauchy_viscoplasticity_VM
   Tutorial_Cosserat_elastoplasticity_VM   
   Tutorial_Cosserat_elastoplasticity_DP
   Tutorial_Cauchy_THM_couplings
   Tutorial Cosserat_THM_thermo_hydro_plasticity - Drucker-Prager yield criterion
   Tutorial_Cosserat_Breakage_Mechanics
   Tutorial_Usage_of_custom_material

