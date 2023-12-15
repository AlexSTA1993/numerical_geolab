.. _Linear and Bilinear forms in Numerical Geolab:

=============================================
Linear and Bilinear forms in Numerical Geolab
=============================================

General formulation of the weak form for a multiphysics coupled solid mechanics problem
=======================================================================================

Numerical Geolab, implements a general incremental variational form for the solid mechanics problems involving generalized continua taking advantage of a generalized Voigt-type notation. Moreover, the variational form provided below takes into account multiphysics couplings and transient terms that can result from balance equations. 

General variational formulation in generalized Voigt form implemented in Numerical Geolab -No transient terms
-------------------------------------------------------------------------------------------------------------
We consider the incremental formulation of a solid mechanics problem in material inelasticity. In such a case the weak form of the force and momentum balance equations in Voigt form to be solved by nGeo are the following:

.. _multiphysics variational formulation:

.. math::

     -\int_\Omega \Delta{g}_{j} {\tilde{\psi}}_{j}d\Omega+\int_\Omega \Delta{f}_i {\tilde{v}}_id\Omega+\int_\Gamma \Delta{t}_i {\tilde{v}}_id\Gamma=0,
      \label{multiphysics variational formulation}

where :math:`{g}_i` is the Voigt vector of generalized stresses, :math:`{\psi}_i` is the Voigt vector of generalized strains, :math:`{f}_i` is the generalized volumic force vector applied on the volume :math:`\Omega` of the body and :math:`{t}_i` is the generalized traction vector applied at the surface :math:`\Gamma` of the domain. We use the :math:`(\tilde{\cdot})` symbol, in order to denote the testfunction fields necessary for deriving the weak form.

We note that this formulation is general enough to contain also the the mass and energy balance diffusion equations at steady state (absence of the transient terms). %We use the symbol $({\cdot})$ to denote the generalized vectors containing the fluxes, gradients and values of the physical quantities that are coupled to the generalized stresses, strains and forces of the mechanical problem respectively. 
We consider the incremental solution of a solid mechanics problem in material inelasticity (see [ZIE]_). During the :math:`k+1` iteration of the $m$ increment, in the case where no transient terms are considered, the general incremental variational form used by nGeo is the following:

.. [ZIE] O. C. Zienkiewicz and R. L. Taylor. The finite element method for solid and structural mechanics. Elsevier, 2005

.. math::

       \int_\Omega\Delta{\psi}_i\;^\text{m}D^{\text{k}+1}_{ij} {\tilde{\psi}}_{j}d\Omega=-\int_\Omega \;^\text{m}{g}^{\text{k}+1}_{j} {\tilde{\psi}}_{j}d\Omega+\int_\Omega \;^\text{m}{f}_i {\tilde{v}}_id\Omega+\int_\Gamma \;^\text{m}{t}_i {\tilde{v}}_id\Gamma,

where, the term of the left hand side contains the new generalized strain increment :math:`\Delta{\psi}_i` and the generalized inelastic 
material matrix :math:`D_{ij}`. The terms in the right correspond to the residual at the k+1  iteration of the m increment. In the residual, 
we take into account the difference between the external generalized volumic and surface forces :math:`{f}_i,\;{t}_i` at the end of the increment 
versus the internal work done by the internal generalized stresses :math:`{g}_i` after the material update algorithm at the k+1 iteration.

In the case of a classical Cauchy continuum, the terms :math:`\Delta {g}_{j}, {\tilde{\psi}}_{j}` correspond to the stress vector,
:math:`{g}_{j}=\left[\sigma_{11},\sigma_{22},\sigma_{33},\sigma_{23},\sigma_{13},\sigma_{12}\right]^T` and the kinematically admissible energy conjugate strain vector. 
:math:`{\psi}_{j}=\left[\varepsilon_{11},\varepsilon_{22},\varepsilon_{33},\varepsilon_{23},\varepsilon_{13},\varepsilon_{12}\right]^T` respectively. Considering the terms of the 
external work :math:`{f}_i,{t}_i`, they correspond to the vectors of volumic forces and surface tractions respectively. The term :math:`{v}_i` corresponds to the displacement vector field.  

Implementation of generalized continua
--------------------------------------

We note that we can apply the weak from of equation :ref:`multiphysics variational formulation` also to the case of generalized continua, i.e continua 
with microstructure (see [GERM]_). More specifically, in a medium with microstructure, we consider that each material point :math:`(P_M)` has 
its own micro structure, where :math:`P_M` is the center of mass. The displacement of a point :math:`P_M'` close to :math:`P_M` can then be described by the Taylor expansion 
of the displacement in a small distance :math:`x'_j` around :math:`P_M`. Thus, we can write:

.. [GERM] P. Germain. The Method of Virtual Power in Continuum Mechanics . Part 2 : Microstructure Author ( s ): P . Germain Source : SIAM Journal on Applied Mathematics , Vol . 25 , No . 3 ( Nov., 1973 ), pp . 556-575. 25(3):556–575, 1973.

.. math::

       {v}'_i={v}_i+\chi_{ij}x'_j+\chi_{ijk}x'_jx'_k+...\;,

where without loss of generality :math:`\chi_{ijk}` can be assumed symmetric w.r.t. the indices :math:`j,k`. The resulting principle of incremental virtual work 
that gives rise to the appropriate balance equations for the nth-order micromorphic continuum is given by: 

.. _multiphysics variational formulation:

.. math::

       &P_{int}+P_{ext,c}+P_{ext,d}=0,\nonumber\\
       &P_{int}=-\int_\Omega \Delta \tau_{ij}v_{i,j}-(\Delta\tau_{[ij]}\chi_{ij}+\Delta\tau_{ijk}\chi_{ijk}+...)+(\Delta b_{ijk}\kappa_{ijk}+\Delta b_{ijkl}\kappa_{ijkl}+...)d\Omega,\nonumber\\
       &P_{ext,d}=\int_\Omega \Delta {f}_i v_i +\Delta\Psi_{ij}\chi_{ij}+\Delta\Psi_{ijk}\chi_{ijk}+...d\Omega\nonumber\\
       &P_{ext,c}=\int_\Gamma \Delta {t}_i {v}_i+\Delta M_{ij}\chi_{ij}+\Delta M_{ijk}\chi_{ijk}+...d\Gamma,\nonumber\\
       &\kappa_{ijk}=\chi_{ij,k},\text{ and } \kappa_{ijkl}=\chi_{ijk,l},\; ...\;,

where the symbol :math:`\cdot_{\cdot,j}` indicates differentiation with respect to the spatial dimension of the problem, :math:`\tau_{[ij]}` denotes the antisymmetric part of the tensor 
:math:`\tau_{ij}`. The quantities :math:`\Psi_{ij},\Psi_{ijk}` are the generalized volumic forces and :math:`M_{ij},M_{ijk}` the generalized traction forces at the boundary of the domain. 
We can rewrite equation :ref:`generalized continua variational formulation` into :ref:`multiphysics variational formulation` by setting:

.. math::

       &{g}_j=\left[\tau_{pq},\tau_{[pq]}, \tau_{pqr},  b_{pqr}, b_{pqrm}, ...\right]^T_{(1\times N_1)},\nonumber\\
       &{\psi}_j=\left[v_{p,q},\chi_{[pq]}, \chi_{pqr},  \kappa_{pqr}, \kappa_{pqrm}, ...\right]^T_{(1\times N_1)},\nonumber\\
       &{f}_i=\left[f_p,\Psi_{pq},\Psi_{pqr},...\right]^T_{(1\times N_2)},\nonumber\\
       &{t}_i=\left[t_p,M_{pq},M_{pqr},...\right]^T_{(1\times N_2)},\nonumber\\
       &{v}_i=\left[v_p,\chi_{pq},\chi_{pqr},...\right]^T_{(1\times N_2)},\nonumber\\
       & j=1,...,N_1\text{ with } N_1=pq\left[(1+r)+rm(1+...)+...\right],\nonumber\\
       & i=1,...,N_2\text{ with } N_2=p\left[(1+q)+rq(1+...)+...\right].

In the case of generalized continua, the terms :math:`\Delta {g}_{j}, {\psi}_{j}` correspond to the generalized stress vector, and the kinematically admissible energy conjugate generalized strain vector 
as presented in [GERM]_ . Likewise the terms of the external work :math:`{f}_i,{t}_i`, they correspond to the vectors of generalized volumic forces and the generalized surface tractions respectively.

When a first order micromorphic (Cosserat) continuum is used, generalized stress and strain asymmetry ensues, and the user needs to populate the generalized stress and strain vectors by the full components 
of the generalized stress, couple stress and strain and curvature quantities. In the 3D case of the equilibrium this leads to a :math:`[36\times 1]` vector for the generalized stresses 
:math:`{g}_i=\left[\tau_{11},\tau_{12},...,\tau_{33},\mu_{11},\mu_{12},...,\mu_{33}\right]^T` and strains :math:`{\psi}_i=\left[\gamma_{11},\gamma_{12},...,\gamma_{33},\kappa_{11},\kappa_{12},...,\kappa_{33}\right]`. 
For the generalized forces, moments and displacements the vector dimensions are :math:`[6\times 1]` containing :math:`{f}_i=\left[f_1,f_2,f_3,c_1,c_2,c_3\right]^T`, :math:`{t}_i=\left[t_1,t_2,t_3,m_1,m_2,m_3\right]^T` and :math:`{v}_i=\left[u_1,u_2,u_3,\omega_1,\omega_2,\omega_3\right]^T`. 

.. _sec: Gen_Formulation_transient:
   
Implementation of transient terms -THMC couplings
-------------------------------------------------

The consideration of multiphysics couplings results in the use of coupled diffusion equations. This is the case, for instance, when THMC 
couplings are implemented. In this case, the energy balance and mass balance equations need to be taken into account for the description of 
the fields of temperature, pore fluid pressure and concentration of chemical quantities. The incremental weak form of the system of the energy, 
mass, and chemical potential balance equations is given by application of the Galerkin procedure for the total system of equilibrium equations:

.. math::

      \frac{1}{\Delta t}\int_{\Omega} \Delta T \tilde{T}d\Omega+\frac{1}{\Delta t}\int_{\Omega} \Delta P \tilde{P}d\Omega &+\frac{1}{\Delta t}\sum_{i=1}^{N}\int_{\Omega}\Delta Q_i\tilde{Q}_id\Omega\nonumber\\
       &-\int_{\Omega}c_{th}\Delta T_{,i} \tilde{T}_{,i}d\Omega-\int_{\Omega}c_{hy}\Delta P_{,i}\tilde{P}_{,i}d\Omega-\sum_{i=1}^{N}\int_{\Omega}c_{qi}\Delta Q_{i,j}\tilde{Q}_{i,j}d\Omega\nonumber\\
       &-\int_{\Omega}C_T\tilde{T}d\Omega-\int_{\Omega}C_P\tilde{P}_{,i}d\Omega-\sum_{i=1}^{N}\int_{\Omega}C_{Qi}\tilde{Q}_{i}d\Omega\nonumber\\
       &-\int_{\Gamma}\Delta Q^{th}_{i}\tilde{T}_{,i}d\Gamma-\int_{\Gamma}\Delta Q^{hy}_i\tilde{P}_{,i}d\Gamma-\sum_{i=1}^{N}\int_{\Omega}\Delta Q^q_{i,j}\tilde{Q}_{i,j}d\Omega\nonumber\\
       &=0,

where :math:`C_T,C_P,C_{Q1},...,C_{Qn}` are the coupling terms between the diffusion equations and the generalized linear and angular momentum balance equations. 
We note here that the coupling terms are problem specific and as such they will be subject to user modification. We note that we can treat the temperature, pore 
fluid pressure and chemical concentration fields as extra unknowns to be appended on the fields of the generalized unknown displacements. 

The right part of the diffusion equations, which shows a differentiation with respect to time of the unknown fields, can be added in the weak form by the multiplication of the vector 
of unknown increments :math:`\Delta\hat{v}_i=\left[\Delta u_1,...,\Delta \omega_3,\Delta T,\Delta P,...,\Delta Q_n\right]^T` by a mapping vector containing the values 0 and 1 depending on the presence of a time derivative in the system of equations. In the above case the mapping vector will take the form :math:`\alpha_i=[0,...,0,1,1,...,1]^T`. 
We complete the numerical calculation of the derivative by dividing with the time increment :math:`\Delta t`. 
The final incremental weak form of the problem is then given by:

.. _multiphysics variational formulation transient:

.. math::

       \frac{1}{\Delta t}\int_\Omega\delta_{jmp}\alpha_m\Delta {v}_{p} {\tilde{v}}_{j}d\Omega-\int_\Omega\Delta {g}_{j} {\tilde{\psi}}_{j}d\Omega+\int_\Omega\Delta {f}_i {\tilde{v}}_id\Omega+\int_\Gamma\Delta {t}_i {\tilde{v}}_id\Gamma=0,

where :math:`\delta_{jmp}` is a third order tensor denoting the Hadamard product, i.e. :math:`\delta_{jmp}=1` when :math:`j=m=p` and 0 otherwise. 
Furthermore, :math:`\alpha_m=[0,...,0,1,1,...,1]^T` is the time derivative mapping vector. In the external forces the volumic and surface fluxes need to also be appended.

Numerical Implementation in nGeo
================================

Implementation of the weak formulation
--------------------------------------

We focus now on the two methods :py:meth:`setVarForm()`, :py:meth:`setVarFormTransient()` that construct the algebraic system to be solved based on the weak form of the 
system of PDEs, and using the UFL symbolic form language. These two methods contain the weak form for the two general cases of problems currently available in Numerical 
Geolab, i.e. the case of quasistatic and transient analyses. In both cases the Voigt form for the unknowns of the problem is used for the construction of the weak form.

.. code-block:: python
 
    def setVarForm(self):
        """
        Set Jacobian and Residual (Voigt form) default version
        """
        n=FacetNormal(self.mesh)
        ds=Measure("ds", domain=self.mesh,subdomain_data = self.boundaries,metadata=self.metadata)
        Jac=inner(dot(self.to_matrix(self.dsde2),self.epsilon2(self.u)),self.epsilon2(self.v))\
                        *dx(metadata=self.metadata)
        Res = -inner(self.sigma2,self.epsilon2(self.v))*dx(metadata=self.metadata)
        for NM in self.NMbcs:
        Res+= dot(NM.ti,self.v)*ds(NM.region_id)
        for NMn in self.NMnbcs:
        Res+=NMn.p*dot(n,as_vector(np.take(self.v,NMn.indices)))*ds(NMn.region_id)
        for RB in self.RBbcs:
        Res+= dot(np.multiply(RB.ks,self.u),self.v)*ds(RB.region_id)

       Jac+=self.feform.setVarFormAdditionalTerms_Jac(self.u,self.Du,self.v,self.svars2,\
                  self.metadata,0.,self.to_matrix(self.dsde2))
       Res+=self.feform.setVarFormAdditionalTerms_Res(self.u,self.Du,self.v,self.svars2,\
                  self.metadata,0.)
       return Jac, Res

.. code-block:: python

    def setVarFormTransient(self):
        """
        Set Jacobian and Residual (Voigt form) default version for transient problems
        """
        n=FacetNormal(self.mesh)
        ds=Measure("ds", subdomain_data = self.boundaries)

        Jac = (1./self.dt)*inner(as_vector(np.multiply(self.dotv_coeffs(),self.u)) , self.v)*dx(metadata=self.metadata)
        Jac+= (1./self.dt)*self.dt*inner(dot(self.to_matrix(self.dsde2), self.epsilon2(self.u)),self.epsilon2(self.v))*dx(metadata=self.metadata)
    
        Res = -(1./self.dt)*inner(as_vector(np.multiply(self.dotv_coeffs(),self.Du)), self.v)\
                     *dx(metadata=self.metadata) 
        Res+= -(1./self.dt)*self.dt*inner(self.sigma2,self.epsilon2(self.v))\
                     *dx(metadata=self.metadata)
        for NM in self.NMbcs:
            Res+= (1./self.dt)*self.dt*dot(NM.ti,self.v)*ds(NM.region_id)
        for NMn in self.NMnbcs:
            Res+= (1.self.dt)*self.dt*NMn.p*dot(n,as_vector(np.take(self.v,NMn.indices)))\
                        *ds(NMn.region_id)
        for RB in self.RBbcs:
            Res+= (1./self.dt)*self.dt*dot(np.multiply(RB.ks,self.u),self.v)*ds(RB.region_id) 
        Jac+=self.feform.setVarFormAdditionalTerms_Jac(self.u,self.Du,self.v,self.svars2,\
                     self.metadata,self.dt,self.to_matrix(self.dsde2))
        Res+=self.feform.setVarFormAdditionalTerms_Res(self.u,self.Du,self.v,self.svars2,\
                     self.metadata,self.dt)
        return Jac, Res

The calculation of the problem's jacobian matrix (variable :py:const:`Jac`) takes places with the help of the automatic differentiation feature of the UFL language. 
More specifically, automatic differentiation takes place in every term containing the integral of functions that have the :py:meth:`{Testfunction()` and :py:meth:`Trialfunction()` as their arguments 
(terms with variables :py:const:`u,v` in the functions presented above. The evaluation of both the linear and bilinear forms of the residual and its jacobian, respectively,
is performed via Gauss quadrature rule, at the Gauss point of each finite element. We obtain access to the Gauss points of the finite element model by specifying the 
:py:const:`metadata` variable inside the integration measure. The quantities describing the stress field (:py:const:`self.sigma2`) and elastoplastic matrix components (:py:const:`self.dsde2`) correspond to vectorfields 
that are evaluated at the Gauss points and the quantities referring to the nodal output of the incremental displacement field, :py:const:`v ` 
and the output of the :py:meth:`Testfunction()`, :py:const:`u`  are evaluated at the nodes of the finite element model and then projected to the Gauss points via the :py:meth:`self.epsilon2()` method.


