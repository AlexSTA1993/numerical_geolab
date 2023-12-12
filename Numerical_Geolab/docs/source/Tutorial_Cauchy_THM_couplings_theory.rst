.. _Tutorial_Cauchy_THM_theory:

====================================================================================================
Application of Thermo- Hydro- Mechanical (THM) couplings in Numerical Geolab: Theoretical background
====================================================================================================

The ease of creating the variational form with FEniCS for the solution of the underlying system 
of partial differential equations allows us to implement a large variety of models describing
different physical systems and their internal processes. One such system is the fault system.
The fault system consists of a thin zone of small thickness :math:`~`1 mm, which accumulates the majority 
pf slip during an earthquke, which we call the fault gouge. The fault gouge is surrounded by a damaged
(bresciated) zone, whose thickness extends to the order of km.

During an earthquake the fault gouge lies under intense shear. This has as a result the production of excessive
heat inside the layer due to yielding of the material. The material of the fault gouge consists of a 
solid skeleton and the pore water fluid. As heat is produced, the difference between the expansivities of the solid and the fluid phase 
(water expands more than the solid skeleton) leads to pore fluid pressure increase inside the fault gouge that reduces the effective confining 
stress of the fault gouge material (see also :ref:`Figure 1<T_P1>`). This phenomenon is also called thermal pressurization. For a geomaterial the decrease in confining stress leads to a decrease in the yielding strength of the 
material and therefore, to frictional softening.

.. _T_P1:

.. figure:: _images/T_P_1.svg
         :height: 200 px
         :width: 400 px
         :alt: alternate text
         :align: center

         Schematic representation of the phenomenon of thermal pressurization.


Frictional softening due to thermal pressurization is one of the main mechanisms that provoke earthquakes (cite references). 
Its evolution is crucial when we are interested in the nucleation of an earthquake and the energy balance during an earthquake phenomenon.
We can model thermal pressurization with the introduction of THM couplings inside the fault gouge domain. Doing so, we couple the 
mechanical behavior of the fault gouge with the energy and mass balance equations.

Strong form of the system of coupled THM equations
==================================================

For a Cauchy continuum, the strong form of the momentum, energy and mass balance equations, for an increment in deformation is given as follows:

.. math::
   :nowrap:
   :label: TP_system_PDEs
   
   \begin{align*}
   &\Delta\sigma_{ij,j}+\Delta f_i=0,\; F(\sigma_{ij}, P)=J_2(\sigma_{ij})+\color{orange}{(\sigma_{ii}+P)\tan\phi}\color{Black}{-cc\; G(\sigma_{ij}, P)=J_2(\sigma_{ij})+(\sigma_{ii}+P)\tan\psi-cc} \\
   &\color{Black}{\frac{\partial T}{\partial t}=c_{th}\frac{\partial^2 T}{\partial x^2}}+\color{Red}{\frac{1}{\rho C}\sigma_{ij}\dot{\varepsilon}^p_{ij}}\\    
   &\color{Black}{\frac{\partial P}{\partial t}=c_{hy}\frac{\partial^2 P}{\partial x^2}}+\color{violet}{\Lambda\frac{\partial T}{\partial t}}+\color{OliveGreen}{\frac{1}{\beta^\star}\frac{\partial \varepsilon_v}{\partial t}}
   \end{align*}

|    
 where :math:`\Delta f_i` is the incremental vector field of volumic forces acting on the structure under consideration. We also provide the form of the Drucker Prager yielding criterion for the description of the elastoplastic behavior of the granular material (:math:`F(\sigma_{ij},P)`).
 Here :math:`J_2(\sigma_{ij})` is the second invariant of the stress tensor and :math:`\tan\phi` is the friction angle of the granular geomaterial.
  In numerical Geolab the positive stresses are the ones that provoke tension in the structure (:math:`\sigma_{ij}>0` in tension). The parameters 
  :math:`c_{th},\;c_{hy}` [:math:`\text{mm}^2\text{/s}`]  are the thermal and hydraulic diffusivity of the fault gouge material respectively. The parameter 
  :math:`\Lambda=\frac{\lambda^\star}{\beta^\star}` is the ratio of the mixture's thermal expansivity (:math:`\lambda^\star`) and its hydraulic compressibility (:math:`\beta^\star`), and controls 
  the pore fluid pressure increase per unit of temperature increase [:math:`\text{MPa/}^\text{o}\text{C}`]. To the terms contributing to the pore fluid pressure increase,
  we need to take into account the pore fluid pressure decrease due to the expansion of the fault gouge material. The material of the fault gouge can increase its volume due to
  temperature expansion and the plastic flow prescribed by the plastic potential :math:`G(\sigma_{ij},\ P)`. For a mature fault gouge, i.e. 
  a granular geomaterial that has reached its critical state, the dilatancy angle :math:`\psi` is equal to zero. Thus only the thermal expansion is taken into account.

Weak form of the system of coupled THM equations
===================================================    
 
As is the case with the examples describing the mechanical behavior of a structure, in order to perform an analysis with FEniCS we need to 
provide the weak form of the above non-linear system of coupled partial differential equations :eq:`TP_system_PDEs`. Following the Galerkin procedure,
applying a test vector field respecting the problem's initial and boundary conditions, with the help of the :py:class:`TestFunction()` class in FEniCs,
performing `integration by parts`_ and applying the `Divergence Theorem`_ theorem:

.. _integration by parts: https://en.wikipedia.org/wiki/Integration_by_parts
.. _Divergence Theorem: https://en.wikipedia.org/wiki/Divergence_theorem
 
.. math::
   :nowrap:
   :label: TP_system_PDEs
   
   \begin{align*}
   &\int_{\Omega}\Delta \sigma_{ij}\tilde{\varepsilon}_{ij}=\int_S \Delta t_i\tilde{u}_idS\\
   &\int_{\Omega}\frac{\partial T}{\partial t}\tilde{T}d\Omega+c_{th}\int_{\Omega}T_{,i}\tilde{T}_{,i}d\Omega-\frac{1}{\rho C}\int_{\Omega}\sigma_{ij}\dot{\varepsilon}^p_{ij}\tilde{T}d\Omega=\int_{S}q^{th}_i\tilde{T}dS\\
   &\int_{\Omega}\frac{\partial P}{\partial t}\tilde{P}d\Omega+c_{hy}\int_{\Omega}P_{,i}\tilde{P}_{,i}d\Omega-\frac{\lambda^\star}{\beta^\star}\int_{\Omega}\frac{\partial T}{\partial t}\tilde{P}d\Omega+\frac{1}{\beta^\star}\int_\Omega\frac{\partial \varepsilon_v}{\partial t}\tilde{P}d\Omega=\int_{S}q^{hy}_i\tilde{P}dS\\
   \end{align*}
 
| 
The unknowns of the weak problem are the incremental displacement components :math:`\Delta U_i`, temperature :math:`\Delta T` and pressure :math:`\Delta P` fields. In our case, due to the nonlinearity in the mechanical 
component of the problem (elastic perfectly plastic material), we will solve numericaly the above nonlinear system, by applying a Newton-Raphson iterative procedure. To do so we need to
define the residual of the algebraic system to be solved and the direction, where it decreases the fastest (we aim for quadratic convergence if possible). The residual definition is given by:

.. math::
   :nowrap:
   :label: Res
   
   \begin{align*}
   &Res=F_{ext}-F_{int}
   \end{align*}
   
where:

.. math::
   :nowrap:
   :label: Fext_Fint
   
   \begin{align*}   
   &F_{ext}=\int_S \Delta t_i\tilde{u}_idS+\int_{S}q^{th}_i\tilde{T}dS+\int_{S}q^{hy}_i\tilde{P}dS\\
   &\begin{aligned}
   F_{int}=&\int_{\Omega}\Delta \sigma_{ij}\tilde{\varepsilon}_{ij}\\
           &+\int_{\Omega}\frac{\partial T}{\partial t}\tilde{T}d\Omega+c_{th}\int_{\Omega}T_{,i}\tilde{T}_{,i}d\Omega-\frac{1}{\rho C}\int_{\Omega}\sigma_{ij}\dot{\varepsilon}^p_{ij}\tilde{T}d\Omega\\
           &+\int_{\Omega}\frac{\partial P}{\partial t}\tilde{P}d\Omega+c_{hy}\int_{\Omega}P_{,i}\tilde{P}_{,i}d\Omega-\frac{\lambda^\star}{\beta^\star}\int_{\Omega}\frac{\partial T}{\partial t}\tilde{P}d\Omega+\frac{1}{\beta^\star}\int_\Omega\frac{\partial \varepsilon_v}{\partial t}\tilde{P}d\Omega
   \end{aligned}
   \end{align*}

The above quantities indicating the internal and external power of the generalized forces are known as linear forms [#]_, In order to minimize the residual :math:`Res` we need to move to the direction oposite to its maximization which is the opposite of
the gradient vector direction, :math:`-\nabla{Res}`. The gradient vector is defined by the diferentiation of the linear form with respect to all 
independent unknowns of the problem namely the incremental displacement components, the temperature and pore fluid pressure fields :math:`\Delta U_i,T,P` respectively.
In what follows, we will assume that the vector of the external forces is independent of the solution of the nonlinear problem (i.e. no follower loads are applied), therefore:

.. math::
   :nowrap:
   :label: Jac_1
   
   \begin{align*}
   Jac=-\nabla Res=\nabla F_{int}
   \end{align*}


For the above formulation the Jacobian of the system is given as follows:

.. math::
   :nowrap:
   :label: Jac_2
   
   \begin{align*}
   \nabla F_{int}&=\frac{\partial F_{int}}{\partial \Delta U_i}\pmb{\Delta \hat{U}_i}+\frac{\partial F_{int}}{\partial T}\pmb{\hat{T}}+\frac{\partial F_{int}}{\partial P}\pmb{\hat{P}}
   \end{align*}

The bold quantities :math:`\pmb{\hat{\left(\cdot{}\right)}}` indicate the unit vectors directions along the orthonormal system of the unknowns.
For ease of notation we apply the operator :math:`\frac{\partial}{\partial X}\left(\cdot{}\right)`, indicating differentiation of the linear form
by each of the unknown quantities. We look first at the power of the internal mechanical forces. Each component of the above mixed (tensor, vector) field is given by:
 
.. math::
   :nowrap:
   :label: Jac_terms_mech
   
   \begin{align*}
   &\frac{\partial}{\partial \Delta X}\int_{\Omega}\Delta \sigma_{ij}\tilde{\varepsilon}_{ij}d\Omega=\frac{\partial}{\partial \Delta U_i}\int_{\Omega}\Delta \sigma_{ij}\tilde{\varepsilon}_{ij}d\Omega+\frac{\partial}{\partial \Delta T}\int_{\Omega}\Delta \sigma_{ij}\tilde{\varepsilon}_{ij}d\Omega+\frac{\partial}{\partial \Delta P}\int_{\Omega}\Delta \sigma_{ij}\tilde{\varepsilon}_{ij}d\Omega
   \end{align*}

At this point we need to emphasize that the quantities :math:`\Delta \sigma_{ij}` and :math:`\tilde{\varepsilon}_{ij}` defined in the above are the total incremental stress and strain for which the momentum balance is defined. Therefore, these quantities need to be analysed to their corresponding mechanical, thermal and 
hydraulic components before the solution of the problem is sought with Numerical Geolab. The following decomposition holds for the total incremental stress and strain:

.. math::
   :nowrap:
   :label: material_def
   
   \begin{align*}
   &\Delta\sigma_{ij}=\Delta \sigma^\star_{ij}-\Delta P\delta_{ij}
   &\varepsilon_{ij}=\varepsilon^\star_{ij}+\alpha\Delta T\delta_{ij},\;\varepsilon^\star_{ij}=\varepsilon^{\star,e}_{ij}+\varepsilon^{\star,p}_{ij}
   \end{align*}

where, :math:`\Delta \sigma^\star_{ij},\;`\varepsilon^\star_{ij}` are the effective stresses and strains developed by the mechanical deformation of the material.
and :math:`\delta_{ij}` is the kronecker delta. We note also that the effective strain can be decoposed to an elastic (:math:`\varepsilon^{\star,e}_{ij}`) and a plastic (:math:`\varepsilon^{\star,p}_{ij}`) component.   

Jacobian terms of the momentum balance equation
-----------------------------------------------

Replacing :eq:`material_def` into :eq:`Jac_terms_mech` the coresponding Jacobian terms are then given by:

.. math::
   :nowrap:
   :label: Jac_terms_mech
   
   \begin{align*}
   \frac{\partial}{\partial \Delta U_i}\int_{\Omega}\left(D^{ep}_{ijkl}\left(\varepsilon_{kl}-\alpha\Delta T\delta_{kl}\right)-\delta_{ij}\Delta P\right)\tilde{\varepsilon}_{ij}d\Omega=&\frac{\partial}{\partial \Delta U_i}\int_{\Omega}D^{ep}_{ijkl}\varepsilon_{kl}\tilde{\varepsilon}_{ij}d\Omega\\
                                                                                                                                                                                          &-\frac{\partial}{\partial \Delta U_i}\int_{\Omega}\alpha \Delta T D^{ep}_{ijkl}\delta_{kl}\tilde{\varepsilon}_{ij}d\Omega\\
                                                                                                                                                                                          &-\frac{\partial}{\partial \Delta U_i}\int_{\Omega}\Delta P \delta_{ij}\tilde{\varepsilon}_{ij}d\Omega\\
   \frac{\partial}{\partial \Delta T}\int_{\Omega}\left(D^{ep}_{ijkl}\left(\varepsilon_{kl}-\alpha\Delta T\delta_{kl}\right)-\delta_{ij}\Delta P\right)\tilde{\varepsilon}_{ij}d\Omega=&\frac{\partial}{\partial \Delta T}\int_{\Omega}D^{ep}_{ijkl}\varepsilon_{kl}\tilde{\varepsilon}_{ij}d\Omega\\
                                                                                                                                                                                          &-\frac{\partial}{\partial \Delta T}\int_{\Omega}\alpha \Delta T D^{ep}_{ijkl}\delta_{kl}\tilde{\varepsilon}_{ij}d\Omega\\
                                                                                                                                                                                          &-\frac{\partial}{\partial \Delta T}\int_{\Omega}\Delta P \delta_{ij}\tilde{\varepsilon}_{ij}d\Omega\\
      \frac{\partial}{\partial \Delta P}\int_{\Omega}\left(D^{ep}_{ijkl}\left(\varepsilon_{kl}-\alpha\Delta T\delta_{kl}\right)-\delta_{ij}\Delta P\right)\tilde{\varepsilon}_{ij}d\Omega=&\frac{\partial}{\partial \Delta P}\int_{\Omega}D^{ep}_{ijkl}\varepsilon_{kl}\tilde{\varepsilon}_{ij}d\Omega\\
                                                                                                                                                                                          &-\frac{\partial}{\partial \Delta P}\int_{\Omega}\alpha \Delta T D^{ep}_{ijkl}\delta_{kl}\tilde{\varepsilon}_{ij}d\Omega\\
                                                                                                                                                                                          &-\frac{\partial}{\partial \Delta P}\int_{\Omega}\Delta P \delta_{ij}\tilde{\varepsilon}_{ij}d\Omega
   \end{align*}

Since the generalised fields are independent of each other only the terms of the solution that are differentiated with themselves survive, and we obtain:

.. math::
   :nowrap:
   :label: Jac_terms_mech_final
   
   \begin{align*}
   \frac{\partial}{\partial \Delta U_i}\int_{\Omega}\left(D^{ep}_{ijkl}\left(\varepsilon_{kl}-\alpha\Delta T\delta_{kl}\right)-\delta_{ij}\Delta P\right)\tilde{\varepsilon}_{ij}d\Omega=&\frac{\partial}{\partial \Delta U_i}\int_{\Omega}D^{ep}_{ijkl}\varepsilon_{kl}\tilde{\varepsilon}_{ij}d\Omega\\
   \frac{\partial}{\partial \Delta T}\int_{\Omega}\left(D^{ep}_{ijkl}\left(\varepsilon_{kl}-\alpha\Delta T\delta_{kl}\right)-\delta_{ij}\Delta P\right)\tilde{\varepsilon}_{ij}d\Omega=&-\frac{\partial}{\partial \Delta T}\int_{\Omega}\alpha \Delta T D^{ep}_{ijkl}\delta_{kl}\tilde{\varepsilon}_{ij}d\Omega\\
   \frac{\partial}{\partial \Delta P}\int_{\Omega}\left(D^{ep}_{ijkl}\left(\varepsilon_{kl}-\alpha\Delta T\delta_{kl}\right)-\delta_{ij}\Delta P\right)\tilde{\varepsilon}_{ij}d\Omega=&-\frac{\partial}{\partial \Delta P}\int_{\Omega}\Delta P \delta_{ij}\tilde{\varepsilon}_{ij}d\Omega
   \end{align*}

The same procedure needs to be followed for the terms in the linear forms of corresponding to the internal power of the generalised forces of the energy and mass balance components of the problem. 

Jacobian terms of the energy balance equation
---------------------------------------------

The Jacobian terms for the energy balance equation are given by differential of the power of internal generalized forces:

.. math::
   :nowrap:
   :label: Jac_terms_energy_final_1
   
   \begin{align*}
   \frac{\partial}{\partial X}\int_{\Omega}\frac{\partial T}{\partial t}\tilde{T}d\Omega &=\frac{\partial}{\partial T}\int_{\Omega}\frac{\partial T}{\partial t}\tilde{T}d\Omega,\\
   c_{th}\frac{\partial}{\partial X}\int_{\Omega}T_{,i}\tilde{T}_{,i}d\Omega &=c_{th}\frac{\partial}{\partial T}\int_{\Omega}T_{,i}\tilde{T}_{,i}d\Omega,\\
   -\frac{1}{\rho C}\frac{\partial}{\partial X}\int_{\Omega}\sigma_{ij}\dot{\varepsilon}^p_{ij}\tilde{T}d\Omega &=-\frac{1}{\rho C}\frac{\partial}{\partial X}\int_{\Omega}D^{ep}_{ijkl}\varepsilon^\star_{kl}\dot{\varepsilon}^{\star,p}_{ij}\tilde{T}d\Omega,\\
   \end{align*}

where:

.. math::
   :nowrap:
   :label: Jac_terms_energy_final_2
   
   \begin{align*}
   -\frac{1}{\rho C}\frac{\partial}{\partial X}\int_{\Omega}D^{ep}_{ijkl}\varepsilon^\star_{kl}\dot{\varepsilon}^{\star,p}_{ij}\tilde{T}d\Omega &=-\frac{1}{\rho C}\frac{\partial}{\partial U_i}\int_{\Omega}D^{ep}_{ijkl}\varepsilon_{kl}\dot{\varepsilon}^{\star,p}_{ij}\tilde{T}d\Omega-\frac{1}{\rho C}\frac{\partial}{\partial \Delta T}\int_{\Omega}\alpha\Delta TD^{ep}_{ijkl}\delta_{kl}\dot{\varepsilon}^{\star,p}_{ij}\tilde{T}d\Omega .
   \end{align*}


Jacobian terms of the mass balance equation
-------------------------------------------

The Jacobian terms for the mass balance equation are given by differential of the power of internal generalized forces:

.. math::
   :nowrap:
   :label: Jac_terms_energy_final
   
   \begin{align*}
   \frac{\partial}{\partial X}\int_{\Omega}\frac{\partial P}{\partial t}\tilde{P}d\Omega &= \frac{\partial}{\partial P}\int_{\Omega}\frac{\partial P}{\partial t}\tilde{P}d\Omega\\
   c_{hy}\frac{\partial}{\partial X}\int_{\Omega}P_{,i}\tilde{P}_{,i}d\Omega &= c_{hy}\frac{\partial}{\partial P}\int_{\Omega}P_{,i}\tilde{P}_{,i}d\Omega\\
   -\frac{\lambda^\star}{\beta^\star}\frac{\partial}{\partial X}\int_{\Omega}\frac{\partial T}{\partial t}\tilde{P}d\Omega &=-\frac{\lambda^\star}{\beta^\star}\frac{\partial}{\partial T}\int_{\Omega}\frac{\partial T}{\partial t}\tilde{P}d\Omega\\
   \frac{1}{\beta^\star}\frac{\partial}{\partial X}\int_\Omega\frac{\partial \varepsilon_v}{\partial t}\tilde{P}d\Omega &= \frac{1}{\beta^\star}\frac{\partial}{\partial U_i}\int_\Omega\frac{\partial \varepsilon_v}{\partial t}\tilde{P}d\Omega
   \end{align*}


.. [#] We use the term power of the generalized forces to refer to the linear form defined by the mechanical component of the momentum balance weak formulation
       (where the terms internal and external power are strictly defined) and the corresponding linear forms of the energy and mass balance components of the coupled problem.  

   