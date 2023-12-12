'''
Created on Aug 6, 2018

@author: Ioannis Stefanou

'''
from dolfin import *

import dolfin as d

from copy import deepcopy
mpi_comm_world=d.MPI.comm_world
#
from ngeoFE.fedefinitions import FEobject
from ngeoFE.solvers import Backward_Euler_Solver
#
import numpy as np

from operator import itemgetter
from math import *

import gc


class General_FEproblem_properties():
    """
    Defines general properties of the FE problem
    """
    def __init__(self):
        pass

class UserFEproblem():
    """
    Defines a user FE problem

    :param feform: finite element variational formulation
    :type feform: FEformulation
    """
    def __init__(self,feform):
        # Get attributes
        self.set_general_properties()
        # Creates MPI environment
        self.comm=mpi_comm_world
        #print("Hello from process", self.comm.Get_rank())
        # Generate mesh and set topology 
        self.mesh,self.subdomains,self.boundaries = self.create_mesh()
        self.cell = self.mesh.ufl_cell().cellname()
        # Get regions
        self.subdomains = self.create_subdomains(self.mesh)
        self.mark_boundaries(self.boundaries)
        # Create material objects
        self.mats=self.set_materials()
        # Dictionary for BC types
        self.BCtype = {
            "DC": 0, #Dirichlet: increment proportionally to step time
            "DC-C": 2, #Dirichlet: set at the beginning and keep constant
            "NM": 1, #Neumann: increment proportionally to step time
            "NM-C": 3, #Neumann: set at the beginning and keep constant
            "RB": 5, #Robin
            "NM-n": 6, #Neumann: normal to the boundary traction (pressure); increment proportionally to step time
            "NM-n-C": 7, #Neumann: normal to the boundary traction (pressure); set at the biginning and keep constant
            }
        # Set BCs
        self.bcs=self.set_bcs()
        # Set FE formulation
        self.feform=feform 
        # Set indices for history output from residual
        self.histories=self.history_output()
        #Set indices for state variables output at the Gauss points #ALEX 19/05/2022
        self.Gausspointsquerry=self.create_Gauss_point_querry_domain(self.mesh)
               
        self.svars_histories=self.history_svars_output()
        # print(self.svars_histories)
        # Creates FE object

        self.large_displacements = self.large_displacements()
        try:
            # If periodic boundary conditions are defined
            self.feobj=UserFEobject(self.mesh,self.feform,self.subdomains,self.boundaries,self.genprops,self.bcs,self.comm,self.pbcs,self.keep_previous,self.large_displacements)
        except AttributeError:   
            self.feobj=UserFEobject(self.mesh,self.feform,self.subdomains,self.boundaries,self.genprops,self.bcs,self.comm,symbolic_histories=self.histories,Gausspointsquerry=self.Gausspointsquerry,symbolic_svars_histories=self.svars_histories,large_displacements=self.large_displacements)
        # Initializes state variables vector
        self.set_initial_conditions()
        # Creates Incremental Solver object
        self.slv=Backward_Euler_Solver(self.feobj, self.mats)


    def set_general_properties(self):
        """
        Set here all the parameters of the problem, except material properties 
        """
        pass

    def set_materials(self):
        """
        Create material objects and set material parameters
        """
        pass

    def create_mesh(self):
        """
        Set mesh
        :return mesh: problem's mesh 
        :return subdomains: problem's regions of interest. E.g. Different materials
        :return self.boundaries: Define problem boundaries
        """
        pass

    def set_bcs(self):
        """
        Set boundary conditions for the user problem / could be replaced by external mesher, e.g. Abaqus, Gmsh...
        Form of the symbolic bcs is a list:
        bcs = [
                | [region_id,[0,[dof],value]]] for Dirichlet incremental
                | [region_id,[1,ti_vector] for Neumann incremental
                | [region_id,[2,[dof],value]]] for Dirichlet instantaneous
                | [region_id,[3,ti_vector] for Neumann instantaneous
                | [region_id,[5,ti_vector] for Robin incremental
                | [region_id,[6,ti_vector] for point load (Neuman) incremental
                | [region_id,[7,ti_vector] for point load (Neuman) instantaneous
                ]
        :return bcs
        :type nested list
        """
        pass

    def set_periodic_bcs(self):
        """
        Set periodic boundary conditions for the user problem
        """
        pass

    def create_subdomains(self,mesh):
        """
        Create subdomains by marking regions. If no subdomains, set subdomain id=0 everywhere. 
        Subdomains define regions with different properties in the problem

        :param mesh: Domain dolfin mesh
        :type mesh: Mesh
        """
        subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
        subdomains.set_all(0) #assigns material/props number 0 everywhere
        return subdomains

    def mark_boundaries(self,boundaries):
        """
        Mark boundary domains if not marked by mesh file

        :param boundaries: dolfin MeshFunction of mesh order ndim-1 marked with boundary ids 
        :type boundaries: MeshFunction
        """
        pass

    def set_initial_conditions(self):
        """
        Initialize state variables vector
        """
        pass

    def history_output(self):
        """
        Used to get output of residual at selected node 
        """
        pass
    
    def create_Gauss_point_querry_domain(self,mesh):
        """
        creates separate Gauss querry domain for extraction of appropriate svars degrees of freedom from FunctionSpace Vsvars
        """
        pass
    
    def history_svars_output(self):
        """
        Used to get output of svars at selected Gauss point 
        """
        pass
    
    def solve(self,file="",silent=False,summary=True):
        """
        Solves the FE problem

        The final solution is contained in feobj

        :param file: output hdmf filename for saving results (default: "" - no output)
        :param silent: messages display
        :param summary: display incrementation summary 
        :type file: string
        :type silent: boolean
        :type summary: boolean
        :return: True if converged
        :rtype: boolean 
        """
        return self.slv.solve(file,silent,summary)

    def plot_me(self):
        """
        Used to plot selected quantities 
        """
        pass

    def large_displacements(self):
        """
        Sets large displacements flag
        """
        self.large_displacements=False
        pass

class BoC():
    """
    Creates a Boundary Condition

    :param symbolic: symbolic, user input form of BCs
    :param BC: Boundary Condition 
    :type symbolic: List
    :type BC: list of dof's for Neumann and Robin or dolfin DirichletBC
    """
    def __init__(self,symbolic,BC):
        """
        Constructor
        """
        self.description=""
        self.type=symbolic[1][0]
        self.target_value=np.array(symbolic[1][2])
        self.dvalue=self.target_value
        self.value=0.
        self.region_id=symbolic[0]
        self.dof=np.array(symbolic[1][1])
        #self.symbolic=symbolic
        self.BC=BC

class UserFEobject(FEobject):
    """
    Creates a user FE formulation

    :param mesh: dolfin mesh
    :type mesh: Mesh
    :param feform: finite element variational formulation
    :type feform: FEformulation
    :param subdomains: list of dolfin subdomains
    :type subdomains: SubDomain
    :param boundaries: dolfin MeshFunction of mesh order ndim-1 marked with boundary ids 
    :type boundaries: MeshFunction
    :param generalprops: general properties
    :type generalprops: General_FEproblem_properties    
    :param symbolic: symbolic, user input form of BCs 
    :type symbolic: list
    :param comm: parallel communicator
    :type comm: d.MPI.comm_world
    :param pbc: periodic boundary conditions for supermaterial
    :type pbc: SuperFEMaterialPeriodicBoundary
    :param keep_previous: Indicates whether to keep previous solution or not
    :type keep_previous: Logical
    :type symbolic_histories: set of dofs whose values need to be saved
    :param symbolic_histories: list of values
    :type large_displacements: Indicates that large displacements are calculated with the ALE method
    :param large_displacements: Logical
    """
    def __init__(self,mesh,feform,subdomains,boundaries,generalprops,symbolic_bcs,comm,pbc=None,keep_previous=False,symbolic_histories=None,Gausspointsquerry=None,symbolic_svars_histories=None,large_displacements=False):

        self.description="put your description"
        self.pbc=pbc
        self.boundaries=boundaries
        self.DCbcs0=[]; self.DCbcs=[]; self.NMbcs=[]; self.RBbcs=[]; self.NMnbcs=[];
#         self.initial=False ####9/9/2019 Alex put it

        if symbolic_bcs!=None: 
            self.symbolic_bcs=sorted(symbolic_bcs,key=itemgetter(1))
        else:
            self.symbolic_bcs=None
        #set general properties
        self.param=generalprops
        super().__init__(mesh, feform, generalprops.p_nsvars, subdomains, comm, pbc, keep_previous)
        #Set history output dofs
        self.symbolic_history=symbolic_histories
        self.set_history_output_indices()
        #Set state variables history output dofs
        self.Gausspointsquerry=Gausspointsquerry
        
        self.symbolic_svars_history=symbolic_svars_histories
        self.set_svars_history_output_indices()
        if large_displacements == True:
            self.large_displacements= True
        else:
            self.large_displacements= False

    def update_mesh(self, mesh, displacement,minus=False):
        """
        Function that utilizes the ALE module of dolfin to update the mesh after each iteration. 
        If the increment in total is not converged then it resets to the previously converged increment.
        :param mesh: dolfin mesh.e    
        :type mesh: Mesh.
        :param displacement: displacement calculated at the specific iteration of the increment/ total displacement of the unconverged increment.
        :type displacement: dolfin Function.
        :param minus: If true resets position of the mesh to previously converged increment.
        :type minus: Logical.
        """
        # For VectorFunctionSpace: works when mesh geometrical dimension (x[:]) is equal to the topological (v[:])
        # V = VectorFunctionSpace(mesh,"CG", 1)
        # disp1= Function(V)
        # v = displacement

        # For MixedFunctionSpace: Works for 1D problems with mixed elements. 
        # Displacements are at the first VectorFunctionSpace
        # For an Interval domain: Only the displacement parallel to the interval is considered, because this includes the volume change 
        V = FunctionSpace(mesh,"CG", 1)
        disp1=Function(V)
        v1,v2 = displacement.sub(0).split(deepcopy=False)

        if minus == False: #when the solver does not converge the applied displacement needs to be removed before the new half increment is applied see the solver
            # v.vector().set_local(v.vector().get_local()) #Suitable for the VectorFunctionSpace definition
            v1.vector().set_local(v1.vector().get_local())#/1000.) #Suitable for the MixedFunctionSpace definition. When scaling is applied this needs to be taken into account
        else:
            # v.vector().set_local(-1.*v.vector().get_local()) #Same as above 
            v1.vector().set_local(-1.*v1.vector().get_local())#/1000.)
        # assign(disp1,v)
        assign(disp1,v1)
        
        ALE.move(mesh, disp1)
        
        return

    def init_Jac_res(self):
        """
        Initialize the Jacobian and the residual through their respective variational forms. 
        Change between transient or quasistatic problems suitable for first derivative in time.
        ::* To do: add inertial variational formulation.
        :ivar dotv_coeffs(): if empty then quasistatic case.
        :var dt: time increment. 
        :type dt: dolfin Expression object.
        :var Jac: Initialized Jacobian object.
        :type Jac: numpy array of reals.
        :var Res: Initialized Residual object.
        :type Res: numpy array of reals.
        """
        #Set variational forms
        if self.dotv_coeffs()!=None:           
            self.dt=Expression("dt",dt=0.,degree=1)
            self.Jac, self.Res = self.setVarFormTransient()
        else:
            self.Jac, self.Res = self.setVarForm()

    def __get_BC_point_or_facet(self,region_id,sV,bc_value):
        """
        Treats point, edge and with surface id BC's
        :param region_id: A list of values. If len(region_id)=1 indicates
             the appropriate boundary for the Dirichlet condition to be applied. Else if len(region_id)=2 then
             A point generalized displacement will be applied.
             region_id[0]= the boundary where the condition will be applied
             region_id[1]= nested list of the position [x1,x2,x3] of the applied displacement  
            The position is given as argument via dolphin syntax inside the DirichletBC dolfin function.
            Alternatively region_id[1] can be given as a string in dolfin syntax.
        :type region_id: nested list 
        :param bc_value: Indicated the value of the Boundary condition
        :type bc_value: real
        :param sV: 
        :type sV: dolfin FunctionSpace.
        """
        if type(region_id) is list:
            meta=region_id[1]
            point_load=""
            if type(meta) is list:
                for i_dim in range(len(meta)):
                    if meta[i_dim]!="all":
                        point_load=point_load+"near(x["+str(i_dim)+"],"+str(meta[i_dim])+") && "
                point_load=point_load[:-4]
            elif type(meta) is str:
                point_load=meta
            BC=DirichletBC(sV, bc_value, point_load, method="pointwise")
        else:
            # BC=DirichletBC(sV, bc_value, self.boundaries, region_id, method="geometric") #Change to work also for discontinuous elements
            BC=DirichletBC(sV, bc_value, self.boundaries, region_id, method="topological") #Change to work also for discontinuous elements

        return BC

    def set_history_output_indices(self):
        """
        Finds the ids of the nodes in the residual marked by the user for history output and saves them to FEobject
        :var symbolic_history: List containing the region and dofs whose history output is needed.
        :type symbolic_history: nested list.
        :var sV: Function subspace of the degrees of freedom needed.
        :type sV: dolfin FunctionSubspace object.
        :var history_indices_ui:
        :var history_indices_ti:
        """    
        if self.symbolic_history==None: return
        #reset at each step
        self.hist_DCbcs=[]; self.hist_NMbcs=[];
        # get dofmap
        dofmap = self.V.dofmap()
        #get local to global dof map        
        lc_to_gl_dof=dofmap.dofs()
        
        self.history_indices_ui=[];
        self.history_indices_ti=[];  

        for i in range(len(self.symbolic_history)):
            region_id=self.symbolic_history[i][0]
            bc_dof=self.symbolic_history[i][1][1]
            bc_value=0.

            if len(bc_dof)==1:
                if self.V.num_sub_spaces()>1:
                    sV=self.V.sub(bc_dof[0])
                else:
                    sV=self.V; bc_value=[bc_value] 
            else:
                sV=self.V.sub(bc_dof[0]).sub(bc_dof[1])
            BC=self.__get_BC_point_or_facet(region_id,sV,bc_value)
            local_dofs_values=BC.get_boundary_values() #returns in local....... perversity  
            # Set Dirichlet Boundary Conditions

            if self.symbolic_history[i][1][0]==0:                                  
# initial position of history_indices_ui=[]
                for lc_dof in local_dofs_values.keys():
                    if lc_dof<=len(lc_to_gl_dof): self.history_indices_ui.append(lc_dof)
            # Set Neumann Boundary Conditions  
# initial position of history_indices_ti=[]d           
            elif self.symbolic_history[i][1][0]==1:

                for lc_dof in local_dofs_values.keys():
                    if lc_dof<=len(lc_to_gl_dof): self.history_indices_ti.append(lc_dof)

    def set_svars_history_output_indices(self):
        """
        Finds the ids of the gauss_points in the state variables vector marked by the user for state variable history output and saves them to FEobject
        :var symbolic_svars_history: List containing the region and dofs whose history output is needed.
        :type symbolic_svars_history: nested list.
        :var sV: Quadrature Function subspace of the degrees of freedom needed in the discontinuous Quadrature space.
        :type sV: dolfin FunctionSubspace object.
        :var svars_history_indices_ui:
        :var svars_history_indices_ti:
        """            
        if self.symbolic_svars_history==None: return
        #reset at each step
        self.svars_hist_DCbcs=[]; self.svars_hist_NMbcs=[];
        gdim = self.mesh.geometry().dim()
        
        self.svars_history_indices=[];
        self.svars_coordinates=[];

        for i in range(len(self.symbolic_svars_history)):
            region_id=self.symbolic_svars_history[i][0]
            bc_dof=self.symbolic_svars_history[i][1][1]
            bc_value=0.

            if len(bc_dof)==1:
                if self.Vsvars.num_sub_spaces()>1:
                    sV=self.Vsvars.sub(bc_dof[0])
                else:
                    sV=self.Vsvars; bc_value=[bc_value] 
            else:
                sV=self.Vsvars.sub(bc_dof[0]).sub(bc_dof[1])
                
            marked_cells = SubsetIterator(self.Gausspointsquerry, region_id)

            for cell_no in marked_cells:
                self.svars_history_indices.append(sV.dofmap().cell_dofs(cell_no.index())[0])
                cell=cells(cell_no)
                for cell1 in cell:
                    self.svars_coordinates.append(sV.element().tabulate_dof_coordinates(cell1)[0])

    def initBCs(self):
        """
        Initialize and set boundary conditions (marks regions and set BC's)
        :ivar symbolic_bcs: bcs defined in set_bcs()
        :ivar DCbcs0: list of Dirichlet boundary conditions given by Dolfin module
        :ivar DCbcs: list of Dirichlet boundary conditions given by Dolfin module has to be reinitialized \
            because SWIG module does not work properly
        :ivar NMbcs: list of Neumann boundary conditions
        :ivar RBbcs: list of Robin boundary conditions
        :ivar NMnbcs:

        For each symbolic boundary condition the routine looks at the form 0f the condition given:
        [region_id,[type,*****]] ***** indicates the code depends on the type of condition to continue

        if type 6,7
        [region_id,[type,bc_indices,bc_value]]
        else:
        [region_id,[type,bc_dof,bc_value]]
        :ivar bc_dof: refers to the local degrees of freedom as defined by the FiniteElement of Dolfin to be inserted in FunctionSpace
            If Mixed Finite Elements are used then len(bc_dof) != 1 and a sublist needs to be filed indicating the local dof to be constrained.
        :type bc_dof: list
        pass the list of values to function __get_BC_point_or_facet(region_id,sV,bc_value)
        """
        
        if self.symbolic_bcs==None: return
        #reset at each step
        self.DCbcs0=[]; self.DCbcs=[]; self.NMbcs=[]; self.RBbcs=[]; self.NMnbcs=[];
        # get dofmap
        for i in range(len(self.symbolic_bcs)):
            if self.symbolic_bcs[i][1][0]==6 or self.symbolic_bcs[i][1][0]==7:
                region_id=self.symbolic_bcs[i][0]
                bc_indices=self.symbolic_bcs[i][1][1]
                bc_value=0.*self.symbolic_bcs[i][1][2]
            else: 
                region_id=self.symbolic_bcs[i][0]
                bc_dof=self.symbolic_bcs[i][1][1]
                bc_value=0.*self.symbolic_bcs[i][1][2]
                if len(bc_dof)==1:
                    if self.V.num_sub_spaces()>1:
                        sV=self.V.sub(bc_dof[0])
                    else:
                        sV=self.V; bc_value=[bc_value] 
                else:
                    sV=self.V.sub(bc_dof[0]).sub(bc_dof[1])
                BC=self.__get_BC_point_or_facet(region_id,sV,bc_value)

            # Set Dirichlet Boundary Conditions
            if self.symbolic_bcs[i][1][0]==0 or self.symbolic_bcs[i][1][0]==2:                                  
                nBC=BoC(self.symbolic_bcs[i],BC)
                self.DCbcs.append(nBC)
                # has to repeat because copy SWIG object does not function
                BC=self.__get_BC_point_or_facet(region_id,sV,bc_value)
                nBC=BoC(self.symbolic_bcs[i],BC)
                self.DCbcs0.append(nBC)
                #print("heloo!",self.DCbcs0)
            # Set Neumann Boundary Conditions     
            elif self.symbolic_bcs[i][1][0]==1 or self.symbolic_bcs[i][1][0]==3:
                NM=BoC(self.symbolic_bcs[i],BC)
                NM.ti=Function(self.V)
                self.NMbcs.append(NM)
            # Set Robin Boundary Conditions     
            elif self.symbolic_bcs[i][1][0]==5:
                RB=BoC(self.symbolic_bcs[i],BC)
                RB.ks=Function(self.V)
                self.RBbcs.append(RB)
            # Set Neumann boundary normal (pressure) Boundary Conditions     
            elif self.symbolic_bcs[i][1][0]==6 or self.symbolic_bcs[i][1][0]==7:
                NMn=BoC(self.symbolic_bcs[i],self.symbolic_bcs[i][1][0])
                NMn.indices=bc_indices
                NMn.p=Expression("value",degree=0, value=0.)
                self.NMnbcs.append(NMn)
        return

    def incrementBCs(self,t_current,t_old,t_init,t_final):
        """
        Increment boundary conditions values

        :param t_current: time at current increment
        :type t_current: double
        :param t_old: time at previous increment
        :type t_old: double
        """ 
        DCbcs=self.DCbcs; NMbcs=self.NMbcs; RBbcs=self.RBbcs; NMnbcs=self.NMnbcs;
        dt=t_current-t_old
        if self.symbolic_bcs==None: return
        # Update Dirichlet BC's      
        for i in range(len(DCbcs)):
            if DCbcs[i].type==0: # Proportional Dirichlet
                DCbcs[i].dvalue=dt*DCbcs[i].target_value/(t_final-t_init)
                DCbcs[i].BC.set_value(Constant(DCbcs[i].dvalue))
            if DCbcs[i].type==2: # Instantaneous Dirichlet
                DCbcs[i].BC.set_value(Constant(DCbcs[i].dvalue))
                DCbcs[i].dvalue=0.
        # Update Neumann BC's
        for NM in NMbcs:
            if NM.type==1:
                NM.value=NM.target_value*(t_current-t_init)/(t_final-t_init)
                #NM.value=NM.value+dt*NM.target_value
                #NM.value=dt*NM.target_value
                #print("dsadasdadasda",NM.value)
                NM.BC.set_value(Constant(NM.value))
            elif NM.type==3:
                NM.value=NM.target_value
                NM.BC.set_value(Constant(NM.value))
                NM.dvalue=0.
            NM.BC.apply(NM.ti.vector())
        # Update Robin BC's
        for RB in RBbcs:
            if RB.type==5:
                RB.BC.set_value(Constant(RB.dvalue))
                RB.dvalue=0.
            RB.BC.apply(NM.ks.vector())
        for NMn in NMnbcs:
            if NMn.type==6:
                NMn.value=NMn.target_value*(t_current-t_init)/(t_final-t_init)
                NMn.p.value=NMn.value
            elif NMn.type==7:
                NMn.value=NMn.target_value
                NMn.p.value=NMn.value
                NMn.dvalue=0.
        return

    def setfi(self,dt):
        """
        Set user's volumic forces and tractor increment

        :param dt: time increment
        :type dt: double
        """
        tmp=dt*np.zeros(self.ndofs)
        self.f.interpolate(Constant(tmp))#=Constant(0.) #TODO 

    def setVarForm(self):
        """
        Set Jacobian and Residual (Voigt form) default version
        """
        n=FacetNormal(self.mesh)
        #
        ds=Measure("ds", domain=self.mesh,subdomain_data = self.boundaries,metadata=self.metadata)
        Jac = inner(dot(self.to_matrix(self.dsde2),self.epsilon2(self.u)),self.epsilon2(self.v))*dx(metadata=self.metadata)
        Res = -inner(self.sigma2,self.epsilon2(self.v))*dx(metadata=self.metadata)
        for NM in self.NMbcs:
            Res+= dot(NM.ti,self.v)*ds(NM.region_id)
        for NMn in self.NMnbcs:
            Res+=NMn.p*dot(n,as_vector(np.take(self.v,NMn.indices)))*ds(NMn.region_id)
        for RB in self.RBbcs:
            Res+= dot(np.multiply(RB.ks,self.u),self.v)*ds(RB.region_id)

        Jac+=self.feform.setVarFormAdditionalTerms_Jac(self.u,self.Du,self.v,self.svars2,self.metadata,0.,self.to_matrix(self.dsde2))
        Res+=self.feform.setVarFormAdditionalTerms_Res(self.u,self.Du,self.v,self.svars2,self.metadata,0.)
        return Jac, Res

    def setVarFormTransient(self):
        """
        Set Jacobian and Residual (Voigt form) default version for transient problems
        """
        n=FacetNormal(self.mesh)
        #print(self.dt.values())
        ds=Measure("ds", subdomain_data = self.boundaries)#,metadata=self.metadata      )

        Jac = (1./self.dt)*inner(as_vector(np.multiply(self.dotv_coeffs(),self.u)) , self.v)*dx(metadata=self.metadata)
        Jac+= (1./self.dt)*self.dt*inner(dot(self.to_matrix(self.dsde2) , self.epsilon2(self.u)),self.epsilon2(self.v))*dx(metadata=self.metadata)
        Res = -(1./self.dt)*inner(as_vector(np.multiply(self.dotv_coeffs(),self.Du)), self.v)*dx(metadata=self.metadata) 

        Res+= -(1./self.dt)*self.dt*inner(self.sigma2,self.epsilon2(self.v))*dx(metadata=self.metadata)

        for NM in self.NMbcs:
            Res+= (1./self.dt)*self.dt*dot(NM.ti,self.v)*ds(NM.region_id)
        for NMn in self.NMnbcs:
            Res+= (1./self.dt)*self.dt*NMn.p*dot(n,as_vector(np.take(self.v,NMn.indices)))*ds(NMn.region_id)
        for RB in self.RBbcs:
            Res+= (1./self.dt)*self.dt*dot(np.multiply(RB.ks,self.u),self.v)*ds(RB.region_id) 

        Jac+=self.feform.setVarFormAdditionalTerms_Jac(self.u,self.Du,self.v,self.svars2,self.metadata,self.dt,self.to_matrix(self.dsde2))
        Res+=self.feform.setVarFormAdditionalTerms_Res(self.u,self.Du,self.v,self.svars2,self.metadata,self.dt)


        return Jac, Res

    def dotv_coeffs(self):
        """
        Get left hand side (lhs) time derivative coefficients
        :return feform.dotv_coeffs(): list of coefficients of lhs time derivative
        :type feform.dotv_coeffs(): method
        """
        return self.feform.dotv_coeffs()

