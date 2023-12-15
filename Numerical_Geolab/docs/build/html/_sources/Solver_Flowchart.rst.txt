.. _Solver_Flowchart:

================
Solver Flowchart
================
Description of the solver object found in :py:mod:`ngeoFE.solvers`

.. image:: _images/image_4.svg
   :alt: Structure of classes
   :target: _images/image_4.svg
   :align: center

Flowchart describing the implemented incremental Backward Euler solver (General incremental solution loop). The first increment of each analysis step starts by applying the total incremental displacement 
:math:`\Delta U` dependent on the time increment :math:`\Delta t` and the total step time :math:`t^f-t^0`. For the first increment, we zero the increment of the unknown vector :math:`u_i`. 
Thus the the first iteration of the vector of its spatial derivatives :math:`\psi^0_i` is also zero. We use values of stress :math:`{}^\text{m-1}g^\text{k+1}_i` and state variables 
:math:`{}^\text{m-1}\chi^\text{k+1}_i` from the previously converged increment and the loading factor math:`\Delta U` for the formation of the residual. The iterative global Newton-Raphson procedure 
for the minimization of the residual happens inside LOOP 2. After the Newton-Raphson procedure has successfully converged a new increment m+1 begins. 
We update the stress Voigt vector, :math:`{}^\text{m+1}g^\text{0}_i`, the state variable vector, :math:`{}^\text{m-1}\chi^\text{0}_i`, the solution vector, :math:`{}^\text{m+1}u_i`, and advance the analysis time 
:math:`{}^\text{m+1}t`. 

.. image:: _images/image_42.svg
   :alt: Structure of classes
   :target: _images/image_42.svg
   :align: center

Flowchart describing the custom implemented Backward Euler solver. Loop 3: The iterative update. After the residual at the current 
iteration has been evaluated and the new iteration, :math:`du_i`, for the unknown vector is found, we update the increment of the 
unknown quantities :math:`\Delta u_i`, thus providing the new increment of the generalized strain vector :math:`\Delta \psi^{t+\Delta t}_i`. 
Next, we insert this together with the state variables at the previous iteration :math:`\chi^{k}_i` to the material algorithm in order to obtain 
the stress and state variables vectors at the current iteration :math:`{}^\text{m}g^{t+\Delta t}_i,{}^\text{m}\chi^{t+\Delta t}_i` and the 
updated material moduli, :math:`{}^\text{m}D^{t+\Delta t}_{ij}`. If the material algorithm has converged successfully and the global Newton-Raphson 
procedure has not reached the iteration limit for the minimization of the residual, we used the updated stress, state variables vectors 
and material moduli, :math:`{}^\text{m}g^\text{k+1}_i,{}^\text{m}\chi^\text{k+1}_i` ,for the construction of the tangent stiffness matrix 
:math:`{}^\text{m}A^\text{k+1}_{ij},{}^\text{m}D^\text{k+1}_{ij}` and the residual vector :math:`{}^\text{m}b^\text{k+1}_{i}`.