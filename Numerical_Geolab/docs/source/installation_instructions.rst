.. _installation-instructions:

=========================
Installation instructions
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
For the user's convenience an image of numerical geolab that bases on FEniCS is available. It can be found in the sources 
numerical_geolab/docker. 

-----------------------------
Code validation via unittests
-----------------------------

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

-------------------------
Reading the documentation
-------------------------

The user can parse the available documentation present in the docs directory, where a list of documented python files exist 
for the construction and solution of different problems in inelasticity involving multiphysics couplings and 
micromorphic continua. The tutorials for inalsticity multiphysics and micromorphic continua are available using the following link  

:ref:`Numerical_Geolab_Tutorials2`

------------------------------------
Formulate and solve a custom problem
------------------------------------
The user can use the available example files in the tutorials as the basis for the construction and solution of a new problem.
      