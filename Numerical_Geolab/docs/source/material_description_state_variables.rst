.. _material description of state variables:

====================
Material description
====================

The table below contains the material properties used by the Numerical Geolab materials. The user can use its own material libraries 
without the need to follow the properties settings described here.

.. raw:: html

   <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
   <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

   <div style="overflow-x: auto; width: 500px;">
      <table style="border-collapse: collapse; table-layout: fixed; width: auto;">
        <colgroup>
            <col style="width: 100px;"> <!-- Adjust the width for the first column -->
            <col style="width: auto;"> <!-- Adjust the width for other columns -->
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
            <col style="width: auto;">
        </colgroup>
        <thead>
            <tr>
                <th>Materials</th>
                <th>p_nsvars</th>
                <th>props[0]</th>
                <th>props[1]</th>
                <th>props[2]</th>
                <th>props[3]</th>
                <th>props[4]</th>
                <th>props[5]</th>
                <th>props[6]</th>
                <th>props[7]</th>
                <th>props[8]</th>
                <th>props[9]</th>
                <th>props[10]</th>
                <th>props[11]</th>
                <th>props[12]</th>
                <th>props[13]</th>
                <th>props[14]</th>
                <th>props[15]</th>
                <th>props[16]</th>
                <th>props[17]</th>
                <th>props[18]</th>
                <th>props[19]</th>
                <th>props[20]</th>
                <th>props[21]</th>
                <th>props[22]</th>
                <th>props[23]</th>
                <th>props[24]</th>
                <th>props[25]</th>
                <th>props[26]</th>
                <th>props[27]</th>
                <th>props[28]</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="position: sticky; left: 0; background-color: white; z-index: 1;">Cauchy3D-DP</td>
                <td>38</td>
                <td>\( K \)</td>
                <td>\( G \)</td>
                <td>NA</td>
                <td>NA</td>
                <td>NA</td>
                <td>NA</td>
                <td>NA</td>
                <td>NA</td>
                <td>NA</td>    
                <td>NA</td>
                <td>\(\tan{\phi}\)</td>
                <td>\(c\)</td>
                <td>\(\tan{\psi}\)</td>
                <td>\(H_{s\phi}\)</td>
                <td>\(H_{sc}\)</td>`
                <td>NA</td>
                <td>NA</td>     
                <td>NA</td>
                <td>NA</td>
                <td>\(\eta\)</td>`
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <!-- Add more cells for each row -->
            </tr>
            <tr>
                <td style="position: sticky; left: 0; background-color: white; z-index: 1;">Cosserat3D-DP</td>
                <td>85</td>
                <td>\( K \)</td>
                <td>\( G \)</td>
                <td>\( Gc \)</td>
                <td>\( L \)</td>
                <td>\( M \)</td>
                <td>\( Mc \)</td>
                <td>NA</td>
                <td>NA</td>    
                <td>NA</td>
                <td>\( R \)</td>
                <td>\(\tan{\phi}\)</td>
                <td>\(c\)</td>
                <td>\(\tan{\psi}\)</td>
                <td>\(H_{s\phi}\)</td>
                <td>\(H_{sc}\)</td>`
                <td>h1</td>
                <td>\(h_2\)</td>     
                <td>\(h_3\)</td>
                <td>\(h_4\)</td>
                <td>\(g_1\)</td>
                <td>\(g_2\)</td>     
                <td>\(g_3\)</td>
                <td>\(g_4\)</td>
                <td>\(\eta\)</td>`
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <!-- Add more cells for each row -->
            </tr>
            <tr>
                <td style="position: sticky; left: 0; background-color: white; z-index: 1;">Cauchy3D-DP-PR-TEMP</td>
                <td>62</td>
                <td>\( K \)</td>
                <td>\( G \)</td>
                <td>\( \chi \)</td>
                <td>\( \eta_f \)</td>
                <td>\(\beta^\star\)</td>
                <td>\(k_T\)</td>
                <td>\(\rho C\)</td>
                <td>\(\alpha\)</td>
                <td>\(\lambda^\star\)</td>    
                <td>NA</td>
                <td>\(\tan{\phi}\)</td>
                <td>\(c\)</td>
                <td>\(\tan{\psi}\)</td>
                <td>\(H_{s\phi}\)</td>
                <td>\(H_{sc}\)</td>`
                <td>NA</td>
                <td>NA</td>     
                <td>NA</td>
                <td>NA</td>
                <td>\(\eta\)</td>`
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <td>OoB</td>
                <!-- Add more cells for each row -->
            </tr>
            <tr>
                <td style="position: sticky; left: 0; background-color: white; z-index: 1;">Cosserat3D-THM</td>
                <td>110</td>
                <td>\( K \)</td>
                <td>\( G \)</td>
                <td>\( Gc \)</td>
                <td>\( L \)</td>
                <td>\( M \)</td>
                <td>\( Mc \)</td>
                <td>\(R\)</td>
                <td>\(\chi\)</td>    
                <td>\(\eta_f\)</td>
                <td>\( \beta^\star \)</td>
                <td>\(k_T\)</td>
                <td>\(\rho C\)</td>    
                <td>\(\alpha\)</td>
                <td>\(\lambda^\star\)</td>
                <td>NA</td>
                <td>\(\tan{\phi}\)</td>
                <td>\(c\)</td>
                <td>\(\tan{\psi}\)</td>
                <td>\(H_{s\phi}\)</td>
                <td>\(H_{sc}\)</td>`
                <td>h1</td>
                <td>\(h_2\)</td>     
                <td>\(h_3\)</td>
                <td>\(h_4\)</td>
                <td>\(g_1\)</td>
                <td>\(g_2\)</td>     
                <td>\(g_3\)</td>
                <td>\(g_4\)</td>
                <td>\(\eta\)</td>`
                <!-- Add more cells for each row -->
            </tr>
            
            <!-- Add more rows as needed -->
        </tbody>
      </table>
   </div>

Explanation of the symbols in the Table:

.. list-table:: Material properties
   :widths: 25 25 50
   :header-rows: 1

   * - Material Property
     - Dimensions
     - Description
   * - :math:`[L]`
     - 
     - Length
   * - :math:`[T]`
     - 
     - Time
   * - :math:`[M]`
     - 
     - mass     
   * - :math:`[K]`
     - 
     - Temperature     
   * - :math:`K`
     - :math:`\frac{[M]}{[T^2][L]}`
     - Triaxial compression modulo
   * - :math:`G`
     - :math:`\frac{[M]}{[T^2][L]}`
     - Shear modulo
   * - :math:`Gc`
     - :math:`\frac{[M]}{[T^2][L]}`
     - Shear modulo owing to anisotropy of Cosserat Material   
   * - :math:`L`
     - :math:`\frac{[M][L]}{[T^2]}`
     - Triaxial Moment modulo for Cosserat Material        
   * - :math:`MG`
     - :math:`\frac{[M][L]}{[T^2]}`
     - Shear Moment modulo for Cosserat Material        
   * - :math:`MGc`
     - :math:`\frac{[M][L]}{[T^2]}`
     - Shear Moment modulo for Cosserat Material             
   * - :math:`R`
     - :math:`[L]`
     - Cosserat radius             
   * - :math:`\chi`
     - :math:`[L^2]`
     - Hydraulic permeability             
   * - :math:`\eta_f`
     - :math:`\frac{[M]}{[T]}`
     - Fluid viscosity    
   * - :math:`\beta^\star`
     - :math:`\frac{[L][T]^2}{[M]}`
     - Hydraulic compressibility         
   * - :math:`k_T`
     - :math:`\frac{[M][L]^2}{[K][T]^2}`
     - Thermal conductivity                            
   * - :math:`\rho C`
     - :math:`\frac{[M]}{[L][K][T]^2}`
     - Specific heat density                            
   * - :math:`\alpha`
     - :math:`\frac{[1]}{[K]}`
     - Thermal expansion coefficient in one dimension                            
   * - :math:`\lambda^\star`
     - :math:`\frac{[1]}{[K]}`
     - Thermal expansivity coefficient
   * - :math:`\phi`
     - :math:`-`
     - Friction angle
   * - :math:`\psi`
     - :math:`-`
     - Dilation angle
   * - :math:`c`
     - :math:`\frac{[M]}{[T^2][L]}`
     - Cohesion
   * - :math:`H_{s\phi}`
     - :math:`-`
     - Hardening/Softening coefficient on the friction angle     
   * - :math:`H_{sc}`
     - :math:`-`
     - Hardening/Softening coefficient on the cohesion term     
   * - :math:`h_1,...,h_4`
     - :math:`-`
     - Cosserat specific parameters owing to the Drucker -Prager yield criterion     
   * - :math:`g_1,...,g_4`
     - :math:`-`
     - Cosserat specific parameters owing to the Drucker -Prager plastic potential    
   * - :math:`NA`
     - :math:`-`
     - Not Attributed        
   * - :math:`OoB`
     - :math:`-`
     - Out of Bounds          

Further description of the material can be found in the following parts of the documentation:      
     
.. toctree::
   :numbered:
   :maxdepth: 2


   CAUCHY3D-DP material

.. todo::   
   COSSERAT3D-DP material
   COSSERAT3D-EXP material
   COSSERAT3D-BREAKAGE material
   CAUCHY3D-DP-PR-TEMP material
   COSSERAT3D-THM material
   COSSERAT3D-DP material   