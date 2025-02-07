'''
Created on Oct 10, 2018

@author: Ioannis Stefanou.
'''
from dolfin import *

import numpy as np

from ngeoFE.feproblem import UserFEproblem, General_FEproblem_properties
from ngeoFE.fedefinitions import FEformulation
from ngeoFE.materials import UserMaterial
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
import time



#from dolfin.cpp.mesh import MeshFunction, IntervalMesh, SubDomain
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

# import time
# import math
# import sys
# from dolfin.cpp.io import HDF5File

class Cauchy_3D_FEformulation(FEformulation):
    '''
    Defines a user FE formulation
    '''

    def __init__(self):
        # Number of stress/deformation components
        self.p_nstr = 6
        # Number of Gauss points
        self.ns = 1

    def generalized_epsilon(self, v):
        """
        Set user's generalized deformation vector
        """
        scale=1./1000.
        gde=[
            Dx(scale*v[0],0),             #gamma_11
            Dx(scale*v[1],1),             #gamma_22
            Dx(scale*v[2],2),             #gamma_22
            Dx(scale*v[1],2)+Dx(scale*v[2],1),   #gamma_23
            Dx(scale*v[0],2)+Dx(scale*v[2],0),   #gamma_13
            Dx(scale*v[0],1)+Dx(scale*v[1],0)   #gamma_12
            ]
        return as_vector(gde)

    def create_element(self, cell):
        """
        Set desired element
        """
        # Defines a Lagrangian FE of degree 1 for the 3 displacements
        #print(cell)
        element = VectorElement("Lagrange", cell, degree=1, dim=3)
        return element


class Triaxial_FEproblem(UserFEproblem):
    """
    Defines a user FE problem for given FE formulation
    """

    def __init__(self, FEformulation):
        self.description = "Triaxial test using Cauchy continuum"
        self.problem_step = 0
        super().__init__(FEformulation)

    def set_general_properties(self):
        """
        Set here all the parameters of the problem, except material properties
        """
        self.genprops = General_FEproblem_properties()
        # Number of state variables
        self.genprops.p_nsvars = 28#26

    def create_mesh(self):
        """
        Set mesh and subdomains
        """
#         pt=Point(0.,0.,100.)
#         pb=Point(0.,0.,0.)
#         geometry=Cylinder(pt,pb,25.,25.)
#         mesh=generate_mesh(geometry,32)
        self.h1=1.
        self.h2=1.
        self.h3=1.
        self.nx=8
        self.ny=8
        self.nz=8
        mesh=BoxMesh(Point(-0.5*self.h1,-0.5*self.h2,-0.5*self.h3),Point(0.5*self.h1,0.5*self.h2,0.5*self.h3),self.nx,self.ny,self.nz)
        import matplotlib.pyplot as plt
        #plot(mesh, title="cubic mesh", wireframe=True)
        #plt.show()
        
      
        cd = MeshFunction("size_t", mesh, mesh.topology().dim())
        fd = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        return mesh, cd, fd
    
    # Define the imperfection
    class Imperfection(SubDomain):
        def inside(self, x, on_boundary):
            return between(x[0], (-.25, .25)) and between(x[1], (-0.25, 0.25)) and between(x[2], (-0.25, 0.25))

    def create_subdomains(self, mesh):
        """
        Create subdomains by marking regions
        """
        subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
        subdomains.set_all(0)  # assigns material/props number 0 everywhere
        imperfection = self.Imperfection()
        imperfection.mark(subdomains, 1)  # assigns material/props number 1 to the imperfection
        return subdomains
    
    class Boundary(SubDomain):
        def __init__(self,xyz,param):
            self.xyz=xyz
            self.param=param
            super().__init__()
        def inside(self, x, on_boundary):
            tol = DOLFIN_EPS
            return on_boundary and near(x[self.xyz],self.param)    
        
    def mark_boundaries(self, boundaries):
        """
        Mark left and right boundary points
        """

        top0 = self.Boundary(2,self.h3/2.)
        top0.mark(boundaries, 1)
        bottom0 = self.Boundary(2,-self.h3/2.)
        bottom0.mark(boundaries, 2)
        #
        left0 = self.Boundary(0,-self.h1/2.)
        left0.mark(boundaries, 3)
        right0 = self.Boundary(0,self.h1/2.)
        right0.mark(boundaries, 4)
        #         
        back0 = self.Boundary(1,-self.h2/2.)
        back0.mark(boundaries, 5)
        front0 = self.Boundary(1,self.h2/2.)
        front0.mark(boundaries, 6)
        
        return

    def set_bcs(self):
        """
        Set boundary conditions for the user problem / could be replaced by external mesher, e.g. Abaqus, Gmsh...
        """
#         if self.problem_step == 0:
        '''
        #displacement controlled triaxial test
        #iso_disp=-0.251
        X_disp=-0.2
        Y_disp=-0.2
        Z_disp=-0.9
        bcs = [
                # [regiod_id,[0,[dof],value]]] for Dirichlet
                # [regiod_id,[1,ti_vector] for Neumann
                # [1,[0, [0,0],u_n/2.]], #delta_u_1=V[0,0]
                #bottom
                [2, [0, [2], 0.]],  
                #top
                [1, [0, [2], Z_disp]],  
                #left
                [3, [0, [0], 0.]],
                #right 
                [4, [0, [0], 0.]], 
                #back
                [5, [0, [1], -0.]],
                #front 
                [6, [0, [1], 0.]], 
                ]
        '''
        scale=1./1000.
        u_displ=-0.16*scale**-1
        
        #         desc2="near(x[2],"+str(self.h3/2.)+")" 
#         desc2b="near(x[2],"+str(-self.h3/2.)+")"
#         desc0="near(x[0],"+str(self.h1/2.)+")"

        bcs = [               
                #top
                [1, [0, [2], u_displ]],
                #bottom
                [2, [0, [2], 0.]],    
                #left
                [3, [0, [0], 0.]],
                #right
                [4, [0, [0], 0.]], 
                #back
                [5, [0, [1], 0.]],
                #front 
                [6, [0, [1], 0.]],
                
            ]        
#         elif self.problem_step == 1:
#             bcs = [
#                 # [regiod_id,[0,[dof],value]]] for Dirichlet
#                 # [regiod_id,[1,ti_vector] for Neumann
#                 # [1,[0, [0,0],u_n/2.]], #delta_u_1=V[0,0]
#                 #top
#                 [1, [0, [2], u_axial/2.]],  
#                 [1, [0, [1], 0.]], 
#                 [1, [0, [0], 0.]],  
#                 #bottom
#                 [2, [0, [2], 0.]],  
#                 [2, [0, [1], 0.]], 
#                 [2, [0, [0], 0.]], 
#                 ]
        return bcs

    def set_materials(self):
        """
        Create material objects and set material parameters
        """
        mats = []
        # load material #1
        env_lib=['/usr/lib/x86_64-linux-gnu/liblapack.so']
        umat_lib='/mnt/c/My_documents/These/Modelling/Fenics/ACC_model/2019-05-14/libplast_Cauchy3D_Asym_CamClay.so'
        umat_id = 3       # if many materials exist in the same library
        mat = UserMaterial(env_lib, umat_lib, umat_id)
        mat.props=self.set_material_1_properties()
        #
        mats.append(mat)
        # load material #2
        env_lib=['/usr/lib/x86_64-linux-gnu/liblapack.so']
        umat_lib='/mnt/c/My_documents/These/Modelling/Fenics/ACC_model/2019-05-14/libplast_Cauchy3D_Asym_CamClay.so'
        umat_id = 3  # if many materials exist in the same library
        mat = UserMaterial(env_lib, umat_lib, umat_id)
        mat.props=self.set_material_2_properties()
        #
        mats.append(mat)
        # load material #3
        # ...
        return mats

    def set_material_1_properties(self):
        """
        Sets material parameters
        """
        # Set material parameters
        EE=20.
        nu=0.16
        GG=EE/(2.*(1.+nu))
        KK=EE*GG/(3*(3*GG-EE))
        pc=-2.
        pt=-2.
        k=-0.94
        M=0.91
        Hev=-1.
        Ha=4.
        pp0=0.#0.0009999999999999996
        props=np.array([KK,GG,0.,0.,0.,0.,0.,0.,k,pt,pc,M,Hev/pc,Ha/pc,0.,0.,0.,0.,pp0])
        props=props.astype("double")
        return props
    
    def set_material_2_properties(self):
        """
        Sets material parameters
        """
        # Set material parameters
        defect_percentage=0.95
        EE=20.
        nu=0.16
        GG=EE/(2.*(1.+nu))
        KK=EE*GG/(3*(3*GG-EE))
        pc=-2.
        pt=-2.
        k=-0.94
        M=0.91
        Hev=-1.
        Ha=4.
        pp0=0.#0.0009999999999999996
        props=np.array([KK,GG,0.,0.,0.,0.,0.,0.,k,pt,pc*defect_percentage,M,Hev/pc,Ha/pc,0.,0.,0.,0.,pp0])
        props=props.astype("double")
        return props
    


my_FEformulation = Cauchy_3D_FEformulation()
my_FEproblem = Triaxial_FEproblem(my_FEformulation)
#saveto="/mnt/f/DEVELOPMENT/1DFORFENICS/res/a_homogeneous.xdmf"
saveto="/mnt/c/My_documents/These/Modelling/Fenics/ACC_model/2019-02-22/ACC_25.xdmf"
#my_FEproblem.slv.dtmax = .2
#print("Consolidating...")

start = time.time() 
my_FEproblem.slv.tmax = 1.
my_FEproblem.slv.dtmax = .3
my_FEproblem.slv.nitermax = 50
my_FEproblem.slv.nincmax = 10000
my_FEproblem.slv.convergence_tol = 1.e-5  # has to be bigger than the materials
converged = my_FEproblem.solve(saveto,summary=False)
if my_FEproblem.feobj.comm.Get_rank()==0: print("Execution time at central (s): ",time.time()-start) 
print("Done")
# change BC's
# my_FEproblem.problem_step = 1
# my_FEproblem.bcs = my_FEproblem.set_bcs()
# my_FEproblem.feobj.symbolic_bcs = sorted(my_FEproblem.bcs, key=itemgetter(1))
# 
# my_FEproblem.slv.tmax = 2.
# my_FEproblem.slv.dtmax = .5
# my_FEproblem.slv.nitermax = 50
# my_FEproblem.slv.nincmax = 10000
# my_FEproblem.slv.convergence_tol = 1.e-5  # has to be bigger than the materials
# print("Shearing...")
# saveto="/mnt/f/DEVELOPMENT/1DFORFENICS/res/test1D_BKG_shearing_noimperf_h10.xdmf"
# converged = my_FEproblem.solve(saveto, summary=False)
