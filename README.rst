.. Numerical Geolab documentation master file, created by
   sphinx-quickstart on Fri Aug 24 17:40:35 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================  
Numerical Geolab's documentation
================================
.. image:: _Numerical_Geolab/docs/source/_images/CoQuake_Banner2.png
   :alt: ERC-CoQuake project logo
   :target: http://www.coquake.com
   :align: center

About
-----

*Numerical Geolab* codes and algorithms serve as a basic ingredient for the numerical developments of the *CoQuake* project.

CoQuake project receives funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 757848 "Controlling earthQuakes").

For more details visit: `CoQuake Project <http://www.coquake.com>`_

Contact: `Ioannis Stefanou <http://istefanou1@gmail.com>`_

The Finite Element module builts on : `FEniCS <https://fenicsproject.org>_` (open-source, under `LGPLv3 <https://www.gnu.org/licenses/lgpl-3.0.en.html>_`). 

*Numerical Geolab (nGeo)* is an open source computational platform for the solution of engineering problems involving material inelasticity, multiphysics, and generalized continua. nGeo
solves the underlying mathematical form incrementally and uses the Voigt-type notation, in order to handle micromorphic continua and multiphysics equations.

For complete description refer to  `Documentation <Numerical_Geolab/docs/source/index>_`

Installation_instructions
=========================

Direct download from source
---------------------------

The project sources including Numerical_Geolab and numerical_geolab_materials are available on Github. 
The user can download the project and add the modules Numerical_Geolab and numerical_geolab_materials to PYTHONPATH i.e:
   
For Linux systems:

.. code-block:: bash

   $ export PYTHONPATH=/path/to/numerical_geolab/Numerical_Geolab
   $ export PYTHONPATH=/path/to/numerical/geolab_materials:$PYTHONPATH

.. Important::
   For the unittests below to work properly: Numerical_Geolab and numerical_geolab_materials should be located under the same directory

Docker container
----------------
For the user's convenience an image of numerical geolab that bases on FEniCS is available. It can be pulled from dockerhub

.. code-block:: bash
   
   $ docker pull alexsta1993/numerical_geolab

Suggested workflow
==================

Runbing the unittests
---------------------

Numerical Geolab comes equipped with unittests that verify the expected behavior for the material libraries, the
mechanical models and also the multiphysics models that use the available materials. In order to test that the files work successfully,  
a script is available that performs all tests. The user execute it as follows:  
for Linux systems:

a) Open a terminal and change the user directory to /path/to/numerical_geolab/Numerical_Geolab/ngeoFE_unittests i.e.

.. code-block:: bash

   $ cd /path/to/numerical_geolab/Numerical_Geolab/ngeoFE_unittests

b) run the python module 0run_all_tests.py i.e.:

.. code-block:: bash

   $ python3 0run_all_tests.py

During the validation procedure numerical geolab can produce plots of specific analysis quantities intrinsic to the unittests inside the ngeoFE/reference_data file
indicating the evolution. These include stress-strain, temperature-pressure, 
plastic strain rate evolution and where applicable relative errors between analytical and numerical solution  during the tests. 

For Linux systems:
The user can activate the plotting option by using the following command:

.. code-block:: bash

   $ export RUN_TESTS_WITH_PLOTS=true


Reading the documentation
-------------------------

The user can parse the available documentation present in the docs directory, where a list of documented python files exist 
for the construction and solution of different problems in inelasticity involving multiphysics couplings and 
micromorphic continua. The tutorials for inalsticity multiphysics and micromorphic continua are available in  

`Documentation <Numerical_Geolab/docs/source/index>_`

Formulating and solving a custom problem
-----------------------------------------

The user can use the available example files in the `Tutorials <Numerical_Geolab/docs/source/index>_` as the basis for the construction and solution of a new problem.

Project structure
=================

.. image:: _Numerical_Geolab/docs/source/images/classes.png
   :alt: Structure of classes
   :target: _Numerical_Geolab/docs/source/images/classes.png
   :align: center

Project modules
...............

Main modules of Numerical Geolab nGeo

* ngeoAI
* ngeoFE
* ngeoFE_unittests


Numerical Geolab Theory
-----------------------
   
Construction of the variational formulation in nGeo

* Linear_and_Bilinear_forms_in_Numerical Geolab


Numerical Geolab Tutorials
==========================

List of available tutials (see also `Documentation<Numerical_Geolab/docs/source/index>_`).

* Tutorial_Cauchy_elastoplasticity_VM
* Tutorial_Cauchy_viscoplasticity_VM
* Tutorial_Cosserat_elastoplasticity_VM   
* Tutorial_Cosserat_elastoplasticity_DP
* Tutorial_Cauchy_THM_couplings
* Tutorial Cosserat_THM_thermo_hydro_plasticity - Drucker-Prager yield criterion
* Tutorial_Cosserat_Breakage_Mechanics
* Tutorial_Usage_of_custom_material



Future contributions
--------------------
* Machine Learning (in progress...)
* Discrete Elements
* Large displacements/deformations (at the moment use of the ALE module available in FEniCS)
* Contact/Interfaces
* Improve accuracy of diffusion in unit-tests (use of a centered finite difference algorithm for the time discretization)
   


