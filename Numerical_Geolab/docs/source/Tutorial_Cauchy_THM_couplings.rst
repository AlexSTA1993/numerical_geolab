Application of Thermo- Hydro- Mechanical (THM) couplings in Numerical Geolab
============================================================================

In this series of tutorials we will study the indroduction of Thermo- Hydro- Mechanical couplings in the Numerical Geolab  software
and by extention into FEniCS. We will study both the theoretical background for the application of the linear form of the residual 
and the bilinear form of the jacobian in order to derive the variational form to be solved by the FEniCs software. We will also provide 
the implementation of the variational form in Numerical Geolab with additional commentary. Finally, we will present a series of 
applications ranging from elasticity to plastity, showcasing the effect of the different couplings in the response of the model.

| Each application is separately described in the documentation

.. toctree::
   :maxdepth: 2
      
   Tutorial_Cauchy_THM_couplings_theory
   Tutorial_Cauchy_THM_couplings_implementation
   Tutorial Cauchy_THM_thermoelasticity
   Tutorial Cauchy_THM_hydroelasticity
   Tutorial Cauchy_THM_thermoplasticity - Von Mises yield criterion
   Tutorial Cauchy_THM_hydroplasticity - Drucker-Prager yield criterion
   Tutorial Cauchy_THM_thermo_hydro_plasticity - Von Mises yield criterion
   Tutorial Cauchy_THM_thermo_hydro_plasticity - Drucker-Prager yield criterion