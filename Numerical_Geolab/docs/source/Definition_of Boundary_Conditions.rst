.. _Definition_of Boundary_Conditions:

================================
Definitionof Boundary Conditions
================================

An important aspect for any finite element model, is the implementation  of the boundary conditions. In the input file script, 
the user provides a nested list specifying a) the region where the boundary condition is to be applied, b) the type of boundary 
condition to use, c) the degree of freedom to constrain and finally, d) the final value of the boundary condition at the end of each step.
This is done inside the :py:meth:`set_bcs()` in the input file. The definition takes the following form: 

.. code-block:: python

        bcs = [[region_id,[bc_type,[dof],value]]]

The arguments of the list are the following:

* :py:const:`region_id`: Region of the boundary where the boundary condition is applied.
* :py:const:`{bc_type}`: Type of the boundary condition specified.
* :py:const:`[dof]`: Degree of freedom affected by the boundary condition.
* :py:const:`value`: Target value of the degree of freedom at the end of the analysis.

The following values for the arguments can then be specified:
The :py:const:`region_id` takes as a value an integer specifying the region of the boundary it is referring to.

The :py:const:`bc_type` takes as a value an integer specifying the type of the boundary condition. 
The user can choose from the following implemented types, namely:

* 0: Dirichlet boundary condition, increasing proportionally to the step time.
* 1: Neumann boundary condition, increasing proportionally to the step time.
* 2: Dirichlet boundary condition, set at the beginning of the step and kept constant.
* 3: Neumann boundary condition, set at the beginning of the step and kept constant.
* 5: Robin boundary condition, set at the beginning of the step and kept constant.
* 6: Neumann boundary condition normal to the boundary surface, increasing proportionally to the analysis step.
* 7: Neumann boundary condition normal to the boundary surface, set at the beginning of the step and kept constant.

The :py:const:`dof` variable takes as a value an integer, specifying the component of the vectorspace, in the framework of FEniCS, 
that will be constrained in the specified :py:const:`region_id`. The vectorspace of the problem is specified by the user during the 
finite element definition. The dictionary containing the map between integer value and type of boundary condition can be found inside 
:py:meth:`UserFEproblem()`.  
