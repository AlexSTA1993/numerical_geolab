'''
Created on Aug 27, 2018

@author: Ioannis Stefanou
'''
from ngeoFE.materials import UserMaterial
# from dolfin.cpp.function import near
from ufl.tensors import as_scalar
from sympy.physics.tests.test_paulialgebra import sigma1

'''
Created on Aug 7, 2018

@author: Ioannis Stefanou
'''
from dolfin import *
import numpy as np
#
from ngeoFE.feproblem import UserFEproblem
from ngeoFE.fedefinitions import FEformulation
#
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
# from dolfin.cpp.io import HDF5File
# from dolfin.cpp.mesh import MeshFunction, SubDomain, UnitSquareMesh

warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)


class SuperFEMaterial():
    """
    Super material class
    """
    def __init__(self,map_type):
        """
        Initialize supermaterial

        .. todo: check if the supermaterial object is stored locally at rank x proccess > It better shouldn't; I think it doesn't.
        """        
        self.silent=True
        self.FEformulation=self.SuperFEMaterialFEformulation()
        self.FEproblems=[]
        self.hom_expr=[]
        self.map_type=map_type

    def get_stress_averages(self,GP_id,stressGP):
        '''
        Calculate average stresses

        .. todo: 
            Have to check if it is properly working in parallel
        '''
        stressGP[:]=assemble(dot(
            self.FEproblems[GP_id].feobj.sigma2
            ,self.FEproblems[GP_id].cR_sigma)*dx).get_local()
        return

    def get_dsde_averages(self,GP_id,dsdeGP):
        '''
        Calculate average stiffness tensor (incremental)

        :param GP_id: id of Gauss point

        .. todo:: Set the correct expression for the C_hom
        '''
        #II=np.identity(3).flatten()
        #print(II)
        dsdeGP[:]=assemble(dot(
            self.FEproblems[GP_id].feobj.dsde2
            ,self.FEproblems[GP_id].cR_dsde)*dx).get_local()

    def plot_state(self,GP_id):
        import matplotlib.pyplot as plt
        #plt.ion()
        plt.xlabel("$x_1$")
        font = {'size'   : 18}
        plt.rc('font', **font)
        Pavg=FunctionSpace(self.FEproblems[GP_id].mesh,"DG",0)
        #p=plot(self.FEproblems[GP_id].feobj.usol[0])#, label='$u_2$')
        plt.ylabel("$\sigma_{11}$")
        p=plot(project(self.FEproblems[GP_id].feobj.sigma2[0],Pavg))#,label='$\sigma_88$')#, label='1')
        #plt.colorbar(p);
        #plt.legend()
        plt.show()
        plt.ylabel("$u_1^{(1)}$")
        #p=plot(project(self.FEproblems[GP_id].feobj.usol[1],Pavg))#,label='$\sigma_88$')#, label='1')
        #print(self.FEproblems[GP_id].feobj.usol.vector().get_local())
        p=plot(self.FEproblems[GP_id].feobj.usol[0])#,label='$\sigma_88$')#, label='1')
        #plt.colorbar(p);
        #plt.legend()
        plt.show()

        #plt.colorbar(p);    

    def usermatGP(self,stressGP_t,deGP,svarsGP_t,dsdeGP_t,dt,GP_id,aux_deGP=np.zeros(1)):
        """
        Super-user material at a Gauss point

        :param GP_id:
        :param stressGP_t:
        :param deGP:
        :param svarsGP_t:
        :param dsdeGP_t:
        :param dt:
        """
        # Solve the auxiliary problem for the increment        
        if GP_id>len(self.FEproblems)-1:
            for i in range(GP_id-len(self.FEproblems)+1):
                self.FEproblems.append(self.SuperFEMaterialFEproblem(self.FEformulation,self.map_type))        

        t=svarsGP_t[0]
        tmaterial=self.FEproblems[GP_id].slv.t
        if tmaterial>t:
            #this means that the solver requests a fresh increment in the material -> roll back to previous state
            self.roll_back_to_previous_state(GP_id)

        if dt!=0:
            self.update_macro_epsilon(deGP,GP_id)
            self.FEproblems[GP_id].slv.set_init_stress=True
            #it will increment -> so save the current state
            self.save_state(GP_id)
        else:
            self.FEproblems[GP_id].slv.set_init_stress=False

        self.FEproblems[GP_id].slv.tmax=t+dt
        converged=self.FEproblems[GP_id].solve("",silent=self.silent)

        if converged==True:
            self.set_output_tdt(GP_id,stressGP_t,dsdeGP_t,svarsGP_t)
            return 0
        else:
            #roll back
            self.roll_back_to_previous_state(GP_id)
            svarsGP_t[0]=self.FEproblems[GP_id].slv.t
            return 1

    def set_output_tdt(self,GP_id,stress_tdt,dsde_tdt,svars_tdt):
        self.get_stress_averages(GP_id,stress_tdt)
        self.get_dsde_averages(GP_id,dsde_tdt) # has to be calculated better
        #other svars may be energies
        svars_tdt[0]=self.FEproblems[GP_id].slv.t

    def save_state(self,GP_id):
        #this might be slow as it involves a lot of copies -> optimization? pointers?
        self.FEproblems[GP_id].slv.t_prev=self.FEproblems[GP_id].slv.t
        #has to be a copy:
        self.FEproblems[GP_id].t_eps_prev=self.FEproblems[GP_id].t_eps.copy()
        self.FEproblems[GP_id].feobj.sigma2_prev.vector().set_local(self.FEproblems[GP_id].feobj.sigma2.vector().get_local())
        self.FEproblems[GP_id].feobj.svars2_prev.vector().set_local(self.FEproblems[GP_id].feobj.svars2.vector().get_local())
        self.FEproblems[GP_id].feobj.dsde2_prev.vector().set_local(self.FEproblems[GP_id].feobj.dsde2.vector().get_local())
        self.FEproblems[GP_id].feobj.usol_prev.vector().set_local(self.FEproblems[GP_id].feobj.usol.vector().get_local())
        return

    def roll_back_to_previous_state(self,GP_id):
        #this might be slow as it involves a lot of copies -> optimization? pointers?
        self.FEproblems[GP_id].slv.t=self.FEproblems[GP_id].slv.t_prev
        #has to be a copy:
        self.FEproblems[GP_id].t_eps=self.FEproblems[GP_id].t_eps_prev.copy()
        self.FEproblems[GP_id].feobj.sigma2.vector().set_local(self.FEproblems[GP_id].feobj.sigma2_prev.vector().get_local())
        self.FEproblems[GP_id].feobj.svars2.vector().set_local(self.FEproblems[GP_id].feobj.svars2_prev.vector().get_local())
        self.FEproblems[GP_id].feobj.dsde2.vector().set_local(self.FEproblems[GP_id].feobj.dsde2_prev.vector().get_local())
        self.FEproblems[GP_id].feobj.usol.vector().set_local(self.FEproblems[GP_id].feobj.usol_prev.vector().get_local())
        return

    def update_macro_epsilon(self,eps,GP_id):
        '''
        Updates generalized strain increment

        :param map_type:
        :param eps:
        :param GP_id:

        .. todo:: 
            * Add diffusion
            * Generalize for 3D Cauchy
            * Add micromorphic
        '''
        self.FEproblems[GP_id].t_eps=eps#np.add(self.FEproblems[GP_id].t_eps,eps)
        #
        if self.map_type=="Cauchy_1D_AUSSOIS_2019":
            self.FEproblems[GP_id].hom_expr.E_00 = self.FEproblems[GP_id].t_eps[0]
            self.FEproblems[GP_id].hom_expr.G_10 = self.FEproblems[GP_id].t_eps[1]            
        elif self.map_type=="Cauchy_2D":
            self.FEproblems[GP_id].hom_expr.E_00 = self.FEproblems[GP_id].t_eps[0]
            self.FEproblems[GP_id].hom_expr.E_11 = self.FEproblems[GP_id].t_eps[1]
            self.FEproblems[GP_id].hom_expr.G_01 = self.FEproblems[GP_id].t_eps[2]
        #self.FEproblems[GP_id].feobj.u0.interpolate(self.FEproblems[GP_id].hom_expr) 
        #self.FEproblems[GP_id].feobj.u0=self.FEproblems[GP_id].hom_expr


    class SuperFEMaterialFEformulation(FEformulation):
        '''
        Defines a user FE formulation for the supermaterial
        '''
        def __init__(self):
            # Number of stress/deformation components
            self.p_nstr=0
            # Number of Gauss points
            self.ns=0

        def generalized_epsilon(self,v):
            """
            Set user's generalized deformation vector
            """
            pass

        def create_element(self,cell):
            """
            Set desired element
            """
            pass  

    class SuperFEMaterialFEproblem(UserFEproblem):
        """
        Defines FE problem for given FE formulation of the supermaterial

        Only orthogonal elementary cells are supported (no need for other geometries now)
        """
        def __init__(self,FEformulation,map_type):
            self.description="supermaterial description, Cauchy continuum"
            self.map_type=map_type
            self.set_general_properties()
            self.periodicityvector,self.boundingbox=self._set_periodicity(self.left_bottom_corner,self.right_top_corner)
            self.pbcs=SuperFEMaterial.SuperFEMaterialPeriodicBoundary(self.periodicityvector,self.boundingbox)
            self.keep_previous=True
            super().__init__(FEformulation)
            self._init_micro_to_macro_mapping()
            tmpRsigma=VectorFunctionSpace(self.feobj.mesh, 'R', 0,dim=self.feobj.p_nstr)
            self.cR_sigma=TestFunction(tmpRsigma)
            tmpRdsde=VectorFunctionSpace(self.feobj.mesh, 'R', 0,dim=self.feobj.p_nstr**2)
            self.cR_dsde=TestFunction(tmpRdsde)
            #self.boundaries=self.create_boundary_subdomains(self.mesh)

        def _init_micro_to_macro_mapping(self):
            '''
            Sets micro to macro mapping for periodic homogenization

            .. todo:: 
                * Add diffusion
                * Generalize for 3D Cauchy
                * Add micromorphic
            '''
            if self.map_type=="Cauchy_1D_AUSSOIS_2019":
                expr= ("E_00*x[0]",".5*G_10*x[0]")
                self.hom_expr = Expression(expr,element=self.feobj.element,E_00=0.,G_10=0.)
                self.t_eps=np.array([0.,0.])
            elif self.map_type=="Cauchy_2D":
                expr= ("E_00*x[0]+.5*G_01*x[1]",".5*G_01*x[0]+E_11*x[1]")
                #expr= ("x[0]","0.")
                self.hom_expr = Expression(expr,element=self.feobj.element,E_00=0.,G_01=0.,E_11=0.)
                self.t_eps=np.array([0.,0.,0.])
            #    
            self.feobj.u0=self.hom_expr
            return

        def _set_periodicity(self,left_bottom_corner,right_top_corner):
            '''
            Get periodicity vector and bounding box 

            :param left_bottom_corner:
            :param right_top_corner:
            :type left_bottom_corner:
            :type right_top_corner:
            :return:
            '''
            # Set bounding box of the mesh (e.g. (e.g. [0., 1.] in 1D, [[0.,0.],[1.,1.]] in 2D, [[0.,0.,0.],[1.,1.,1.]] in 3D)
            boundingbox=SuperFEMaterial.Bounding_box([left_bottom_corner,right_top_corner])
            # Set periodicity vector (e.g. [1.] in 1D, [1.,1.] in 2D, [1.,1.,1.] in 3D)
            periodicityvector=list(np.array(right_top_corner) - np.array(left_bottom_corner))
            return periodicityvector,boundingbox

        class pin_point(SubDomain):
            def __init__(self,point):
                self.pt=point
                super().__init__()
            def inside(self, x, on_boundary):
                for i in range(len(self.pt)):
                    a=+(x[i]-self.pt[i])**2
                a=sqrt(a)
                return a<=DOLFIN_EPS_LARGE and on_boundary

        def mark_boundaries(self,boundaries):
            """
            Mark right top corner boundary point with id=10
            """
            pp0 = self.pin_point(self.right_top_corner)
            pp0.mark(boundaries, 10)
            return

        def set_bcs(self):
            """
            Pin point right top corner dof's equalt to zero

            .. todo:: 
                * Add micromorphic
            """

            if "Cauchy" in self.map_type:
                bcs = [
                        [10, [0, [0], 0.]],  
                        [10, [0, [1], 0.]], 
                        [10, [0, [2], 0.]],   
                        ]
                bcs=bcs[0:len(self.right_top_corner)]
            return bcs

        def set_general_properties(self):
            """
            Set here all the parameters of the problem, except material properties 

            When overidden, has to define the maximum number of the state variables of the materials of the supermaterial by setting: self.genprops.p_nsvars= ??
            """
            pass

        def create_mesh(self):
            """
            Set mesh and subdomains

            :return: mesh object
            :rtype: Mesh
            """
            pass

        def set_materials(self):
            """
            Create material objects and set material parameters

            :return: Returns a list of UserMaterial objects
            :rtype: UserMaterial
            """
            pass         

    class Bounding_box():
        '''
        Bounding box object
        '''
        def __init__(self,points):
            '''
            :param points: list that contains the two corners of the bounding box:
             [[left_x,bottom_y,foreground_z],[right_x,top_y,background_z]]
             :type points: List
            '''
            self.l_x=points[0][0]
            self.r_x=points[1][0]
            if len(points[0])==1: return
            self.b_y=points[0][1]
            self.t_y=points[1][1]
            if len(points[0])==2: return
            self.f_z=points[0][2]
            self.b_z=points[1][2]

    class SuperFEMaterialPeriodicBoundary(SubDomain):
        '''
        Defines periodic boundary conditions for orthogonal periodic cell

        I give credit to: Garth N. Wells see `FEniCS Q&A <https://fenicsproject.org/qa/262/possible-specify-more-than-one-periodic-boundary-condition/>`_
        '''
        def __init__(self,periodicityvector,boundingbox):
            SubDomain.__init__(self)
            self.a = periodicityvector
            self.bb = boundingbox
            self.Lx=self.a[0]
            if len(periodicityvector)==1: return
            self.Ly=self.a[1]
            if len(periodicityvector)==2: return
            self.Lz=self.a[2]

        def inside(self, x, on_boundary):
            '''
            Check if point belongs to target domain for the map function. Left, bottom and foreground boundaries are the "target domains"

            :param x: coordinates of the point
            :param on_boundary: True if point belongs to boundary
            :return: True if it belongs to the target domain
            :rtype: boolean 
            '''
            # return True if on left boundary
            if len(self.a)==1:
                return bool(near(x[0], self.bb.l_x) and on_boundary)
            # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
            elif len(self.a)==2:    
                return bool(  (near(x[0], self.bb.l_x) or near(x[1], self.bb.b_y)) and 
                        (not( (near(x[0], self.bb.l_x) and near(x[1], self.bb.t_y)) or 
                              (near(x[0], self.bb.r_x) and near(x[1], self.bb.b_y)) ) ) and on_boundary)
            # return True if on left, bottom or foreground boundary AND NOT on one of the three corners (0,1,0), (1,0,0) and (0,0,1)
            elif len(self.a)==3:
                return bool(  (near(x[0], self.bb.l_x) or near(x[1], self.bb.b_y) or near(x[2], self.bb.f_z)) and 
                        (not( (near(x[0], self.bb.l_x) and near(x[1], self.bb.t_y) and near(x[2], self.bb.f_z)) or 
                              (near(x[0], self.bb.r_x) and near(x[1], self.bb.b_y) and near(x[2], self.bb.f_z)) or
                              (near(x[0], self.bb.l_x) and near(x[1], self.bb.b_y) and near(x[2], self.bb.b_z)) ) ) and on_boundary)

        def map(self, x, y):
            '''
            Override FEniCS method to map dof's at point x to point y

            :param x: source coordinates
            :param y: target coordinates
            '''
            if len(self.a)==1:
                if near(x[0], self.bb.r_x):
                    y[0] = x[0] - self.a[0]
            elif len(self.a)==2:
                #return if it is on (1,1) corner
                if near(x[0], self.bb.r_x) and near(x[1], self.bb.t_y):
                    y[0] = x[0] - self.a[0]
                    y[1] = x[1] - self.a[1]
                #return if it is on the right boudary
                elif near(x[0], self.bb.r_x):
                    y[0] = x[0] - self.a[0]
                    y[1] = x[1]
                #return if it is on the top boudary
                elif near(x[1], self.bb.t_y):
                    y[0] = x[0]
                    y[1] = x[1] - self.a[1]
            elif len(self.a)==3:
                #return if it is on (1,1,1) corner
                if near(x[0], self.bb.r_x) and near(x[1], self.bb.t_y) and near(x[2], self.bb.b_z):
                    y[0] = x[0] - self.a[0]
                    y[1] = x[1] - self.a[1]
                    y[2] = x[2] - self.a[2]
                #return if it is on the right boudary
                elif near(x[0], self.bb.r_x):
                    y[0] = x[0] - self.a[0]
                    y[1] = x[1]
                    y[2] = x[2]
                #return if it is on the top boudary
                elif near(x[1], self.bb.t_y):
                    y[0] = x[0]
                    y[1] = x[1] - self.a[1]
                    y[2] = x[2]
                #return if it is on the top boudary
                elif near(x[2], self.bb.b_z):
                    y[0] = x[0]
                    y[1] = x[1]
                    y[2] = x[2] - self.a[2]




#                 # Plot solution
#             import matplotlib.pyplot as plt
#              
#             #x = np.linspace(-.5, .5, 25)
#             #y = np.linspace(-1., 1., 50)
#             #X, Y = np.meshgrid(x, y)
#             #Z=
#             #plt.contour(X, Y, Z, 20, cmap='RdGy');
#             plt.xlabel("$x_1$")
#             plt.ylabel("$x_2$")
# #             
#             plot(self.FEproblems[GP_id].mesh)
#             p=plot(self.FEproblems[GP_id].feobj.usol[0], title="$u_1$")
#             #p.set_cmap('RdGy')
#             #p.set_clim(0.0, 1.0)
#             plt.colorbar(p);
#             plt.show()
#             #plt.savefig("demo.png")
#              
#             plot(self.FEproblems[GP_id].mesh)
#             p=plot(self.FEproblems[GP_id].feobj.usol[1], title="$u_2$")
#             #p.set_cmap('RdGy')
#             #p.set_clim(0.0, 1.0)
#             plt.colorbar(p);
#             plt.show()
#             #plt.savefig("demo.png")
#              
#              
#             #plot(my_FEproblem.mesh)
#             Pavg=FunctionSpace(self.FEproblems[GP_id].mesh,"DG",0)
#             p=plot(project(self.FEproblems[GP_id].feobj.sigma2[0],Pavg))#, title="$\sigma_{22}$")
#             print(self.FEproblems[GP_id].feobj.sigma2.vector().get_local())
#             #p.set_cmap('RdGy')
#             p.set_clim(0.0, .4)
#             plt.colorbar(p);
#             plt.show()
#             #plt.savefig("demo.png")
#             #plot(my_FEproblem.mesh)
#             Pavg=FunctionSpace(self.FEproblems[GP_id].mesh,"DG",0)
#             p=plot(project(self.FEproblems[GP_id].feobj.sigma2[1],Pavg))#, title="$\sigma_{22}$")
#             print(self.FEproblems[GP_id].feobj.sigma2.vector().get_local())
#             #p.set_cmap('RdGy')
#             p.set_clim(0.0, .4)
#             plt.colorbar(p);
#             plt.show()
#             #plt.savefig("demo.png")


#         def create_boundary_subdomains(self,mesh):
#             """
#             Mark points for imposing BC to avoid rigid body motions
#             """
#             subdomains = MeshFunction("size_t", mesh, 1)
#             subdomains.set_all(0) #assigns material/props number 0 everywhere
#             _corner_1=self._corner_1(self.periodicityvector,self.boundingbox)
#             _corner_1.mark(subdomains, 1) #assigns material/props number 1 to corner 1
#             if len(self.periodicityvector)==1: return subdomains
#             _corner_2=self._corner_2(self.periodicityvector,self.boundingbox)
#             _corner_2.mark(subdomains, 2) #assigns material/props number 2 to corner 2
#             if len(self.periodicityvector)==2: return subdomains 
#             _corner_3=self._corner_3(self.periodicityvector,self.boundingbox)
#             _corner_3.mark(subdomains, 3) #assigns material/props number 3 to corner 3
#             return subdomains
#
#         class _corner_1(SubDomain):
#             """
#             Gets left, bottom, foreground corner
#             """
#             def __init__(self,periodicityvector,boundingbox):
#                 SubDomain.__init__(self)
#                 self.periodicityvector = periodicityvector
#                 self.boundingbox = boundingbox
#                 
#             def inside(self, x, on_boundary):
#                 if len(self.periodicityvector)==1:
#                     return bool(near(x[0], self.boundingbox.l_x))
#                 elif len(self.periodicityvector)==2:
#                     return bool(near(x[0], self.boundingbox.l_x) and near(x[1], self.boundingbox.b_y)) 
#                 elif len(self.periodicityvector==3):
#                     return bool(near(x[0], self.boundingbox.l_x) and near(x[1], self.boundingbox.b_y) and near(x[2], self.boundingbox.f_z))
#         
#         class _corner_2(SubDomain):
#             """
#             Gets right, bottom, foreground corner
#             """
#             def __init__(self,periodicityvector,boundingbox):
#                 SubDomain.__init__(self)
#                 self.periodicityvector = periodicityvector
#                 self.boundingbox = boundingbox
# 
#             def inside(self, x, on_boundary):
#                 if len(self.periodicityvector)==2:
#                     return bool(near(x[0], self.boundingbox.r_x) and near(x[1], self.boundingbox.b_y)) 
#                 elif len(self.periodicityvector)==3:
#                     return bool(near(x[0], self.boundingbox.r_x) and near(x[1], self.boundingbox.b_y) and near(x[2], self.boundingbox.f_z))
#         
#         class _corner_3(SubDomain):
#             """
#             Gets right, bottom, background corner
#             """
#             def __init__(self,periodicityvector,boundingbox):
#                 SubDomain.__init__(self)
#                 self.periodicityvector = periodicityvector
#                 self.boundingbox = boundingbox
#             
#             def inside(self, x, on_boundary): 
#                 if len(self.periodicityvector)==3:
#                     return bool(near(x[0], self.boundingbox.r_x) and near(x[1], self.boundingbox.b_y) and near(x[2], self.boundingbox.b_z))

#         def set_bcs(self):
#             """
#             Set boundary conditions to avoid rigid body movements (elementary cell is cosidered as Cauchy)
#             """
#             #[topology_id,[dof,value]]
#             if len(self.periodicityvector)==1:
#                 bcs=[[1,[0,0.]]]
#             elif len(self.periodicityvector)==2:
#                 bcs=[
#                     [1,[0,0.]],
#                     [1,[1,0.]],
#                     [2,[1,0.]]
#                     ]
#             elif len(self.periodicityvector)==3:
#                 bcs=[
#                     [1,[0,0.]],
#                     [1,[1,0.]],
#                     [1,[2,0.]],
#                     [2,[1,0.]],
#                     [3,[1,0.]]
#                     ]
#             bcs=[]
#             return bcs    

#             meshfile="/mnt/f/DEVELOPMENT/GMSHFENICS/test2D_1.h5"
#             mesh = Mesh()
#             hdf = HDF5File(mesh.mpi_comm(), meshfile, "r")
#             hdf.read(mesh, "/mesh", False)
#             #cd = CellFunction("size_t", mesh) #deprecated
#             cd=MeshFunction("size_t", mesh, mesh.topology().dim())
#             hdf.read(cd, "/cd")
#             #fd = FacetFunction("size_t", mesh) #deprecated
#             fd=MeshFunction("size_t", mesh, mesh.topology().dim()-1)
#             hdf.read(fd, "/fd")
