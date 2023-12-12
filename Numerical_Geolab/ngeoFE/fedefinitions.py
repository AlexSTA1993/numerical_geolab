"""
Created on Aug 2, 2018

@author: Ioannis Stefanou
"""

from dolfin import *

#from dolfin.cpp.common import set_log_level
#
import numpy as np

class FEformulation():
    """
    Defines general properties of the FE problem
    """
    def __init__(self):
        # Number of stress/deformation components
        self.p_nstr=1
        # Number of Gauss points
        self.ns=1
        # Number of auxiliary quantities at gauss points
        self.p_aux=self.ndofs

    def generalized_epsilon(self,v):
        """
        Set user's generalized deformation vector

        :param v: test function
        :type v: TestFunction
        """
        pass

    def auxiliary_fields(self,v):
        """
        Set user's generalized deformation vector

        :param v: test function
        :type v: TestFunction
        """
        return as_vector(v)

    def create_element(self,cell):
        """
        Set desired element

        :param cell: mesh cell type 
        :type cell: Cell
        """
        pass

    def dotv_coeffs(self):
        """    
        Set left hand side derivative coefficients
        """
        pass 

    def setVarFormAdditionalTerms_Jac(self,u,Du,v,svars,metadata,dt,dsde):
        """
        Set user's additional terms at variational form for Jacobian

        :param u: trial function
        :type u: TrialFunction
        :param Du: increment of trial function
        :type Du: Function
        :param v: test of test function
        :type v: TestFunction
        :param svars: function of state variables
        :type v: svars2
        :param dt: time increment
        :type dt: double
        :param dt: time increment
        :type dt: double

        WARNING: the derivatives for calculating the Jacobian are made with the Trial function u and not with the Function Du.
        """
        Jac=0 
        return Jac

    def setVarFormAdditionalTerms_Res(self,u,Du,v,svars,metadata,dt):
        """
        Set user's additional terms at variational form for Residual

        :param u: trial function
        :type u: TrialFunction
        :param Du: increment of trial function
        :type Du: Function
        :param v: test of test function
        :type v: TestFunction
        :param svars: function of state variables
        :type v: svars2
        :param dt: time increment
        :type dt: double
        """
        Res=0 
        return Res

class FEobject():
    """
    Creates the FE object to be solved with solvers

    :param mesh: dolfin mesh
    :type mesh: Mesh
    :param feform: finite element variational formulation
    :type feform: FEformulation
    :param p_nsvars: number of components of state variables vector
    :type p_nsvars: integer
    :param subdomains: list of dolfin subdomains
    :type subdomains: SubDomain
    :param comm: parallel communicator
    :type comm: d.MPI.comm_world
    :param pbc: periodic boundary conditions for supermaterial
    :type pbc: SuperFEMaterialPeriodicBoundary 
    """
    def __init__(self,mesh,feform,p_nsvars,subdomains,comm,pbc=None,keep_previous=False):
        """
        Constructor

        Attributes:
            self.p_nstr (int): Number of components of the generalized stress vector\n
            self.p_nsvars (int): Number of components of state variables\n
            self.feform (FeFormulation):attribute of the finite element variational formulation (in problems of pure eleasticity it corresponds to the kinematical description of the strains)\n
            self.ns (FeFormulation.ns):attribute of number of Gauss points\n
            self.mesh (UserFeProblem.create_mesh()):attribute of the mesh\n
            self.cell (mesh.ufl_cell().cellname()):indicates the topological dimension of the element used e.g "interval","triangle","tetrahedron"\n
            self.element (self.feform.create_element(self.cell): creates the element with given topology, dimension, interpolation functions and integration order\n
            self.V (FunctionSpace(mesh, self.element, constrained_domain=self.pbc)): Indicates the Function space: It is understood as an arbitrary function derived from the interpolation functions of all the elements assigned in mesh.\n 
            It takes into account local connections of the elements at common nodes and assigns the interpolation coefficients to be defined in the solution. The optional parameter constrained_domain is used for problems with periodic boundary conditions.\n 
            In this case the degrees of freedom at opposite edges of the boundary are set to be common.\n
            self.V0 (FunctionSpace(mesh, self.element)): FunctionSpace used for an initial non-periodic field.\n
            self.v (TestFunction(self.V)): Test function of the variational formulation. \n
            self.p_aux (self.feform.p_aux): Get number of auxilary fields defined on the variational formulation of the element.\n           
            self.u (TrialFunction(self.V)): Define trial functions (unknown generalized displacements) \n         
            self.f (Function(self.V)): Define external generalized volumic forces and tractor \n
            self.u0 (Function(self.V0)): Define initial non-periodic generalized displacements \n
            self.Du (Function(self.V)): Define incremental solution \n
            self.du (Function(self.V)): Define iterative solution \n
            self.usol (Function(self.V, name="Gen_Diplacements")): Define total solution. \n
            self.usol.interpolate(Constant(np.zeros(self.ndofs))):Initialize total solution to zero. \n
             __Ve (VectorElement("Quadrature", self.cell, degree=self.ns,dim=self.feform.p_nstr,quad_scheme='default')): Define a Vector Element whose interpolation function just adds the values of the corresponding stress component at the Gauss points.\n
             For the material definition and subsequent analysis Voight-notation was used taking the components of the stress tensor as vectorial components.  \n
            self.Vstress = FunctionSpace(mesh,__Ve): Create the appropriate FunctionSpace. Due to the elements used ("Quadrature") the global interpolation function is Dirac discontinuous over each  Gauss point and each element such that the value of the prescribed integral\n
            in the value of the integrand itself. Therefore each element is characterized by number (ns) of different stress vectors -one per Gauss point- which add together for the numerical evaluation of the solution integral in the V FunctionSpace.\n
            __Ve (VectorElement("Quadrature", self.cell, degree=self.ns,dim=self.p_aux,quad_scheme='default')): Same as before for the evaluation of the auxillary fields.\n
            self.Vaux  (FunctionSpace(mesh,__Ve)): same as before
            __Ve (VectorElement("Quadrature", self.cell, degree=self.ns,dim=p_nsvars,quad_scheme='default')): Same as before for the evaluation of the state variables at the Gauss points as the material algorithm calculates the material state response\n
            on the Gauss points and numerical integration is performed using the values at the Gauss points. State variables are needed for the calculation of the internal force vector as well as the Jacobian matrix.\n
            self.Vsvars (FunctionSpace(mesh,__Ve)): Same as before.\n
            __Ve = VectorElement("Quadrature", self.cell, degree=self.ns,dim=self.feform.p_nstr*self.feform.p_nstr,quad_scheme='default') The material tangent modulus concerning the generalised stress train response of the material.\n
            self.Vdsde (FunctionSpace(mesh,__Ve)): same as before.\n

            self.sigma2 (Function(self.Vstress)): Function of the current stress state living in the Vstress FunctionSpace.  \n
            self.deGP2 (Function(self.Vstress)): Function of the current strain state living in the Vstress FunctionSpace.\n
            self.aux_deGP2 (Function(self.Vaux)): Function of the auxilary fields spatial directional derivatives as they were defined in FeFormulation,\n
            living in the Vaux FunctionSpace.\n
            self.dsde2 (Function(self.Vdsde)): Function of the material tangent modulus living in the Vdsde FunctionSpace.\n
            \n
            For some problems we need to keep the previous state (e.g. SuperMaterials)\n
            self.keep_previous (logical): Logical flag indicating wether the previous converged state needs to be kept. Important in problems of multiple scales were the Finite-element/material problem at the Gauss points might not converge for all Gauss points in the super element\n
            In this case we need to prevent the converged Gauss points from updating their state as they would normally do.\n
            self.sigma2_prev (Function(self.Vstress)): previous stress vector of last total converged increment. \n
            self.svars2_prev (Function(self.Vsvars)): previous state variables vector of last total converged increment. \n
            self.dsde2_prev (Function(self.Vdsde)): previous material tangent modulus vector of last total converged increment. \n
            self.usol_prev (Function(self.V)): converged solution at the Gauss point at the previous increment from which the incremental procedure starts at the sub problem. \n

            self.metadata (str): Fenics-ufl flag it indicates to the program that access to the Gauss points in needed. Namely it instructs dolfin to perform numerical Gauss integration for calculating the integrals (Nowadays other algorithms may be more efficient especially when material non-linearity is not encountered)\n
            Values used for integration at the Gauss points: {"quadrature_degree":self.ns,"quadrature_scheme":"default"}\n

            self.comm=comm\n

            self.history_indices_ti (int): Indicates the degrees of freedom for which we wish to separately save their force output (e.g traction on a boundary node).  
            self.history_indices_ui (int): Indicates the degrees of freedom for which we wish to separately save their solution output (e.g displacement on a boundary node)
            self.problem_history (list): List gathering the history output of the selected dofs.\n

            self.domainidGP (self.__init_domains(mesh,subdomains,self.ns)): List of the problems Gauss points for every subdomain defined.
        """
        # FEniCS parameters
        #set_log_level(INFO)
        set_log_level(20)
        parameters["form_compiler"]["optimize"] = True
#         parameters["form_compiler"]["cpp_optimize_flags"] = "-O3" # optimization flags for the C++ compiler
        # compatibility with current version only for 3D
        if mesh.ufl_cell().cellname()=="tetrahedron" or mesh.ufl_cell().cellname()=="hexahedron":
            parameters["form_compiler"]["representation"] = 'quadrature'
        else:
            parameters["form_compiler"]["representation"] = 'quadrature'
        self.p_nstr=feform.p_nstr #: No of components of the generalized stress vector
        self.p_nsvars=p_nsvars #: No of components of state variables
        # Set generalized vector function
        self.feform=feform #attribute of the kinematical formulation
        # Set number of Gauss points
        self.ns=feform.ns #:attribute of number of Gauss points
        # Set mesh
        self.mesh=mesh #:attribute of the mesh
        self.cell=mesh.ufl_cell().cellname()  #:indicates the topological dimension of the element used e.g "interval","triangle","tetrahedron"
        # Create element 
        self.element=self.feform.create_element(self.cell) #:creates the element with given topology, dimension, interpolation functions and integration order
        # Assign the element to the mesh
        self.V=FunctionSpace(mesh, self.element, constrained_domain=self.pbc) #:Indicates the Function space: It is understood as an arbitrary function derived from the interpolation functions of all the elements assigned in mesh.\ 
        #:It takes into account local connections of the elements at common nodes and assigns the interpolation coefficients to be defined in the solution. 
        #: Keep a non-periodic space for the u0 (initial displacements)
        self.V0=FunctionSpace(mesh, self.element) #:FunctionSpace used for an initial non-periodic field.\n
        # Define test functions (virtual velocities)
        self.v=TestFunction(self.V) #:Test function of the variational formulation. \n
        # Get number of degrees of freedom
        self.ndofs=np.shape(self.v)[0]
        # Get number of auxiliary fields
        if hasattr(self.feform, 'p_aux')==False: self.feform.p_aux=self.ndofs
        self.p_aux=self.feform.p_aux
        # Define trial functions (unknown generalized displacements)           
        self.u=TrialFunction(self.V)          
        # Define external generalized volumic forces and tractor 
        self.f=Function(self.V)
#         self.tn=[]
        # Define initial non-periodic generalized displacements
        self.u0=Function(self.V0)
        # Define solution increments and solution vectors
        self.Du=Function(self.V)
        self.du=Function(self.V)
        self.usol=Function(self.V, name="Gen_Diplacements")
        # Initialize solution to zero 
        self.usol.interpolate(Constant(np.zeros(self.ndofs)))
        #
        __Ve = VectorElement("Quadrature", self.cell, degree=self.ns,dim=self.feform.p_nstr,quad_scheme='default') #P_NSTR components
        self.Vstress = FunctionSpace(mesh,__Ve)
        __Ve = VectorElement("Quadrature", self.cell, degree=self.ns,dim=self.p_aux,quad_scheme='default') #P_AUX components
        self.Vaux = FunctionSpace(mesh,__Ve)
        __Ve = VectorElement("Quadrature", self.cell, degree=self.ns,dim=p_nsvars,quad_scheme='default') #P_NSVARS components
        self.Vsvars = FunctionSpace(mesh,__Ve)
        __Ve = VectorElement("Quadrature", self.cell, degree=self.ns,dim=self.feform.p_nstr*self.feform.p_nstr,quad_scheme='default') #P_NSTR**2 components
        self.Vdsde = FunctionSpace(mesh,__Ve)
        #
        self.sigma2 = Function(self.Vstress)
        self.deGP2 = Function(self.Vstress)
        self.aux_deGP2 = Function(self.Vaux)
        self.svars2 = Function(self.Vsvars)
        self.dsde2 = Function(self.Vdsde)
        # for problems we need to keep the previous state (e.g. SuperMaterials)
        self.keep_previous=keep_previous
        if self.keep_previous==True:
            self.sigma2_prev = Function(self.Vstress) #:previous stress matrixof last total converged increment
            self.svars2_prev = Function(self.Vsvars)
            self.dsde2_prev = Function(self.Vdsde)
            self.usol_prev = Function(self.V)
        #
        self.metadata={"quadrature_degree":self.ns,"quadrature_scheme":"default"}
        #
        self.comm=comm
        #
        self.history_indices_ti=None
        self.history_indices_ui=None
        self.problem_history=[]

        self.svars_history_indices=None
        self.problem_svars_history=[]

#         #
#         if self.dotv_coeffs()!=None:           
#             self.dt=Expression("dt",dt=0.,degree=1)
#             self.Jac, self.Res = self.setVarFormTransient()
#         else:
#             self.Jac, self.Res = self.setVarForm()            
        #
        self.domainidGP = self.__init_domains(mesh,subdomains,self.ns)

    def set_dt(self,dt):
        """
        Sets new dt to be consider in setVarForm_x

        :param dt: time increment
        :type dt: double
        """
        try:
            self.dt.dt=dt
        except AttributeError:
            return 1
        return 0

    def __init_domains(self,mesh,subdomains,ns):
        """
        Initializes subdomains for setting diffderent material properties (hidden)

        :param mesh: dolfin mesh
        :type mesh: Mesh
        :param subdomains: list of dolfin subdomains
        :type subdomains: SubDomain
        :param ns: polynomial degree for determining the number of Gauss points  
        :type ns: integer
        :return domainidGP: list containing the material id in each Gauss point 
        :rtype: domainidGP.astype("int")
        """

        # Load domain ids to scalar functionspace
        domainid  = Function(FunctionSpace(mesh, 'DG', 0)) #: Load domain ids to scalar functionspace
        #temp = np.asarray(subdomains.array(), dtype=np.int32)
        #domainid.vector()[:] = np.choose(temp, np.arange(len(props)))
        domainid.vector()[:] = np.asarray(subdomains.array(), dtype=np.int32)
        # Space for id of domain and assignement of domain id at GPs
        Ve = FiniteElement("Quadrature", self.cell, degree=ns,quad_scheme='default') #:Create the finite element with discontinuous interpolation functions whose nodes are at the Gauss points. 
        Vdomain = FunctionSpace(mesh,Ve) #: Apply the Quadrature element to the whole of the domain
        domain2=Function(Vdomain) #: A function that specifies the material of the particular Gauss point
        self.local_project(domainid, Vdomain, domain2) #: Projection of the material ids defined at the nodes of each element to its Gauss points.
        domainidGP=domain2.vector().get_local() #: store the material label of each Gauss point in a vector  
        return domainidGP.astype("int")      

    def local_project(self,v,V,u=None): #:V is the function space to project on, v is the funtion to be projected, u is the projected function
        """
        General projection function (used later for calculating values @ GPs

        :param v: dolfin function
        :type v: Function
        :param V: function space
        :type V: FunctionSpace

        I give credit to Jeremy Bleyer for this method (`source <https://comet-fenics.readthedocs.io/en/latest/tips_and_tricks.html#efficient-projection-on-dg-or-quadrature-spaces>`_)
        """
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv,v_)*dx(metadata=self.metadata)
        b_proj = inner(v,v_)*dx(metadata=self.metadata)
        solver = LocalSolver(a_proj,b_proj)
        solver.factorize()
        if u is None:
            u = Function(V)
            solver.solve_local_rhs(u)
            return u
        else:
            solver.solve_local_rhs(u)
            return

    def to_matrix(self,comp_dsde): #probably not optimal, but as it is compiled it doesn't matter
        """
        Convert to matrix

        :param comp_dsde: jacobian ds/de in vector form
        :type comp_dsde: numpy array
        :return: jacobian ds/de in dolfin matrix form
        :rtype: as_matrix
        """
        len1=sqrt(np.shape(comp_dsde)[0])
        if len1 % 1 == 0.:
            len1=int(len1)
            a = [[comp_dsde[i+j*len1] for i in range(len1)] for j in range(len1)]
            return as_matrix(a)
        else:
            print("error: not square matrix.")
            return 0.

    def epsilon2(self,v):
        """
        Get generalized deformation vector

        :param v: dolfin test function
        :type v: TestFunction
        :return: generalized epsilon
        :rtype: feform.generalized_epsilon
        """
        return self.feform.generalized_epsilon(v)

    def aux_field2(self,v):
        """
        Get auxiliary fields for the Gauss points

        :param v: dolfin test function
        :type v: TestFunction
        :return: auxiliary fields
        :rtype: feform.auxiliary_fields
        """
        return self.feform.auxiliary_fields(v)


    def setVarForm(self):
        """
        Set Jacobian and Residual (Voigt form)
        """
        pass

    def setVarFormTransient(self):
        """
        Set Jacobian and Residual (Voigt form)
        """
        pass

    def dotv_coeffs(self):
        """
        Get left hand side time derivative coefficients
        """
        pass







