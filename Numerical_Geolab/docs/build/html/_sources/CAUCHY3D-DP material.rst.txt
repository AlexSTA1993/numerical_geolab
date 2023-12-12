CAUCHY3D-DP
===========

Elastoplastic description
-------------------------
In the undeformed configuration under the assumption of small displacements, the constitutive law of an isotorpic linear elastic material 
is given by the following relation in indicial notation (i=1,...,3, and  summation over repeated indices is taken into account):

.. math::
	\begin{align}
	\begin{aligned}
	&\sigma_{ij}=K\varepsilon^e_v\delta_{ij}+Ge^e_{ij},\\
	&\varepsilon_{v}=\varepsilon_{kk},\; e_{ij}=\varepsilon_{ij}-\varepsilon_v\delta_{ij},\\
	\end{aligned}
	\end{align}
			
where :math:`\sigma_{ij},\;\varepsilon_{ij}`, are the second order stress and strain tensors respectively and :math:`K,G` are the material moduli corresponding to triaxial compression and shear.
From the decomposition of the stress and strain second order tensors into shperical and deviatoric part we obtain the following relations:

.. math::
	\begin{align}
	\begin{aligned}
	&\sigma_{ij}=p\delta_{ij}+s_{ij},\\
	&p=\frac{\sigma_{kk}}{3},\;s_{ij}=\sigma_{ij}-p\delta_{ij}\\
	&p=3 K \varepsilon^e_{v}\\
	&s_{ij} = G \varepsilon^e_{ij}
	\end{aligned}
	\end{align}
The elastic constitutive description in Voigt formulation is given as follows:

.. math::
	\begin{align}
	\begin{bmatrix}
	\sigma_{11}\\
	\sigma_{22}\\
	\sigma_{33}\\
	\sigma_{23}\\
	\sigma_{13}\\
	\sigma_{12}
	\end{bmatrix}
	=
	\begin{bmatrix}
	\frac{4G}{3}.+K	& -\frac{2G}{3}.+K &-\frac{2G}{3}.+K &0 &0 &0\\
	-\frac{2G}{3}.+K	& \frac{4G}{3}.+K  &-\frac{2G}{3}.+K &0 &0 &0\\
	-\frac{2G}{3}.+K	& -\frac{2G}{3}.+K &\frac{4G}{3}.+K  &0 &0 &0\\
					0	&	0			&0					 &G &0 &0\\
					0	&	0			&0					 &0 &G &0\\
					0	&	0			&0					 &0 &0 &G
	\end{bmatrix}
	\begin{bmatrix}
	\varepsilon_{11}\\
	\varepsilon_{22}\\
	\varepsilon_{33}\\
	\gamma_{23}\\
	\gamma_{13}\\
	\gamma_{12}
	\end{bmatrix},
	\end{align}

where :math:`\gamma_{ij}=\varepsilon_{ij},\;i=j` and :math:`\gamma_{ij}=\frac{1}{2}\left(\varepsilon_{ij}+\varepsilon_{ji}\right),\;i\neq j`. The yield criterion is given by the following relation:

.. math::
	\begin{align}
	\begin{aligned}
		&F(\sigma_{ij},e^p_q)=q-\tan \phi p - c(1+h_c e^p_q),\\
		&q=\sqrt{\frac{1}{2}s_{ij}s_{ij}}\\
		&\varepsilon_{ij}=\dot{\lambda}\frac{\partial G}{\partial \sigma_{ij}},\\
		&\varepsilon^p_q=\sqrt{2e^p_{ij}e^p_{ij}},\\
		&\varepsilon^p_v=\dot{\lambda}\frac{\partial G}{\partial p},\\
		&e^p_{ij} =\dot{\lambda}\left(\delta_{mi}\delta_{nj}-\frac{1}{3}\delta_{ij}\delta_{mn}\right)\frac{\partial F}{\partial s_{mn}}.
	\end{aligned}
	\end{align}

The relationship between the plastic strain rate and the applied stresses satisfying the yield criterion involes the plastic multiplier :math:`\lambda`. 
In this relation G is the plastic potential function given by:

.. math::
	\begin{align}
	\begin{aligned}
		&G(\sigma_{ij},e^p_q)=q-\tan \psi p - c(1+h_c e^p_q).
	\end{aligned}
	\end{align}

When associative plasticity is used we have that :math:`F=G` and the Saint-Venant's coaxiality assumption (normality rule) ensues.

In the case of elasto-viscoplasticity the viscoplastic multiplier is given by :math:`\dot{\lambda}=\left(\frac{<G>}{\eta^{vp}}\right)^N` with :math:`N=1`. 
Finally, we note that :math:`<G>=G` when :math:`G>0` and zero otherwise. For an associative material :math:`(F=G)` according to [PON2002]_ the viscoplastic potential is then given by:

.. math::
	\begin{align}
	\Omega(\sigma_{ij},e^p_q)=q-\tan\phi p - c(1+h_ce^p_q)-\eta^{vp}\dot{\lambda}=0.
	\end{align}

The consistency condition then reads:

.. math::
	\Delta {\Omega} = \frac{\partial F}{\partial s_{ij}} \Delta s_{ij}+\frac{\partial F}{\partial p} \Delta p+\frac{\partial F}{\partial e^p_q}\Delta e^p_q+\frac{\partial \Omega}{\partial \dot{\lambda}}\Delta\dot{\lambda}=0

where

.. math::
	\begin{align}
	\begin{aligned}
	&\Delta p=3 K \left(\Delta \varepsilon_{v}-\Delta \varepsilon^p_{v}\right)= 3 K \left(\Delta \varepsilon_{v}-\Delta \lambda \frac{\partial G}{\partial p}\right)\\
	&\Delta s_{ij} = G \left(\Delta e_{ij}-\Delta e^p_{ij}\right)=G \left(\Delta e_{ij}-\Delta \lambda \left(\delta_{mi}\delta_{nj}-\frac{1}{3}\delta_{ij}\delta_{mn}\right)\frac{\partial G}{\partial s_{mn}}\right)\\
	&\Delta e^p_q = \Delta {\lambda}\sqrt{\left(\delta_{mi}\delta_{nj}-\frac{1}{3}\delta_{ij}\delta_{mn}\right)\frac{\partial G}{\partial s_{mn}}\left(\delta_{pi}\delta_{qj}-\frac{1}{3}\delta_{ij}\delta_{pq}\right)\frac{\partial G}{\partial s_{pq}}}\\
	&\Delta\dot{\lambda} = \frac{\Delta <F>}{\eta^{vp}}
	\end{aligned}
	\end{align}

solving for :math:`\Delta \lambda` we obtain:

.. math::
   \begin{align}
   \begin{aligned}
   \Delta \lambda =\frac{2G\frac{\partial F}{\partial \sigma_{ij}}\Delta e_{ij}+\frac{\partial F}{\partial p}K\Delta \varepsilon_v}{2G\left(\delta_{mi}\delta_{nj}-\frac{1}{3}\delta_{ij}\delta_{mn}\right)\frac{\partial F}{\partial s_{ij}}\frac{\partial G}{\partial s_{mn}}+K\frac{\partial F}{\partial p}\frac{\partial G}{\partial p}}+\frac{\eta^{vp}\Delta \dot{\lambda}}{2G\left(\delta_{mi}\delta_{nj}-\frac{1}{3}\delta_{ij}\delta_{mn}\right)\frac{\partial F}{\partial s_{ij}}\frac{\partial G}{\partial s_{mn}}+K\frac{\partial F}{\partial p}\frac{\partial G}{\partial p}}
   \end{aligned}  
   \end{align}

The incremental elastoplastic constitutive description in Voigt formulation is given by solving numerically for :math:`\Delta\lambda` that satisfies the consistency condition and the incremental stress strain relation.
Then the material algorithm has converged to a new state :math:`\sigma^{t+dt}_{ij},\varepsilon^{t+dt}_{ij},\varepsilon^{e,t+dt}_{ij},\varepsilon^{p,t+dt}_{ij}, F,\Delta{\lambda},\Delta\varepsilon^{p,t+dt}_{ij}`.


Material properties
-------------------
For the three dimensional Cauchy Drucker-Prager material the following material properties need to be specified:

.. csv-table:: CAUCHY3D-DP: material properties
   :file: ./_csvfiles/CAUCHY3D-DP/CAUCHY3D-DP_props.csv
   :widths: 30, 30, 30
   :header-rows: 1

Mapping of the accessible state variables
-----------------------------------------
The number of state variables and their meaning are defined in the following table.

.. csv-table:: CAUCHY3D-DP: Mapping of the available state variables
   :file: ./_csvfiles/CAUCHY3D-DP/CAUCHY3D-DP.csv
   :widths: 70, 70, 70,30,70,30
   :header-rows: 1
   
.. [PON2002] Ponthot, J. P. (2002). Unified stress update algorithms for the numerical simulation of large deformation elasto-plastic and elasto-viscoplastic processes. International Journal of Plasticity, 18(1), 91-126.
   
