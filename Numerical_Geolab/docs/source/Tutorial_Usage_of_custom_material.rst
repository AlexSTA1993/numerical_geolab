.. _Tutorial_Usage_of_custom_material:

========================
Usage of custom material
========================

Because the incremental Variational from is calculated at the Gauss points of the problem, Numerical Geolab provides access to the generalized stress and strain Voigt vectors.
These quantities and the increment of the generalized strain are accessible as vector arguments to the user material method, :py:mod:`ngeoFE.materials.UserMaterial.usermatGP`.

Connection between the material library and the python interface is handled by the :py:mod:`~python.ctypes` module of python.

The user needs to verify that the arguments provided in this method agree with the arguments of the custom material shared object library .so.
This can be done by establishing a correspondence between the arguments of the :py:mod:`~ngeoFE.materials.UserMaterial.usermatGP` and the custom user material
inside the custom user material subroutine.

  
