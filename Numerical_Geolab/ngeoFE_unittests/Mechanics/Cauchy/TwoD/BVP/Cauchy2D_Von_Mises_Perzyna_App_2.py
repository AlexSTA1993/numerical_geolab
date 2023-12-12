'''
Created on Nov 05, 2018

@author: Alexandros Stathas
'''
import sys

# sys.path.insert(0,'/home/alexandrosstathas/eclipse-workspace/eclise-virtualenv/lib/python3.8/site-packages/')
# import matplotlib

from dolfin import *
import time
import numpy as np
from ngeoFE.feproblem import UserFEproblem, General_FEproblem_properties
from ngeoFE.fedefinitions import FEformulation
from ngeoFE.materials import UserMaterial
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from dolfin.cpp.io import HDF5File
from sympy.sets.tests.test_sets import test_union_boundary_of_joining_sets
from ngeoFE_unittests import ngeo_parameters
from ngeoFE_unittests import plotting_params 
# ngeo_parameters.reference_data_path='/home/alexandrosstathas/eclipse-workspace/numerical_geolab/Numerical_Geolab/ngeoFE_unittests/Mechanics/reference_data/'
import os
from _operator import itemgetter

# import sys
# print(sys.version)
# print(sys.path)

# import os
# import inspect
# import dolfin
# print(inspect.getfile(dolfin))

#from dolfin.cpp.mesh import MeshFunction
# from dolfin.cpp.mesh import MeshFunction

warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

class Cauchy2DFEformulation(FEformulation):
    '''
    Defines a user FE formulation
    '''
    def __init__(self):
        # Number of stress/deformation components
        self.p_nstr=3
        # Number of Gauss points
        self.ns=1
    
    def generalized_epsilon(self,v):
        """
        Set user's generalized deformation vector
        """
        gde=[
            Dx(v[0],0),  #gamma_11
            Dx(v[1],1),  #gamma_11
            Dx(v[0],1)+Dx(v[1],0)  #gamma_12
            ]
        return as_vector(gde)
    
    def create_element(self,cell):
        """
        Set desired element
        """
        # Defines a Lagrangian FE of degree 1 for the displacements
        element_disp=VectorElement("Lagrange",cell,degree=1,dim=2)
        return element_disp  

class left(SubDomain):
    def inside(self,x,on_boundary):
        return x[0] < 0  and on_boundary

class right(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0 and on_boundary

class top(SubDomain):
    def inside(self,x,on_boundary):
        return x[1] > 0  and on_boundary

class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < 0 and on_boundary

class Gauss_point_Querry(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= 1./2.-1./80. and between(x[1], (-0.1,0.1))
# class Imperfection(SubDomain):
#     """
#     Sets an imperfection
#     """
#     def inside(self, x, on_boundary):
#         return between(x[0], (-0.1, 0.1))
           
class Cauchy2DFEproblem(UserFEproblem):
    """
    Defines a user FE problem for given FE formulation
    """
    def __init__(self,FEformulation):
        self.description="Example of 2D plane strain problem, Cauchy continuum"
        self.problem_step=0
        self.h = 1.
        self.w = 0.2
        super().__init__(FEformulation)  
    
    def set_general_properties(self):
        """
        Set here all the parameters of the problem, except material properties 
        """
        self.genprops=General_FEproblem_properties()
        # Number of state variables
        self.genprops.p_nsvars=38
    
    def create_mesh(self):
        """
        Set mesh and subdomains 
        """
        # Generate mesh
        #mesh=IntervalMesh(self.genprops.ny,-self.genprops.h/2.,self.genprops.h/2.)
        h=self.h
        w=self.w
        ny=80
        nw=1
        mesh = RectangleMesh(Point(-h/2.,-w/2.),Point(h/2.,w/2.),ny,nw,"left")#"crossed")
        #print(mesh.topology().dim())
        cd=MeshFunction("size_t", mesh, mesh.topology().dim())
        fd=MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        return mesh,cd,fd

    def create_subdomains(self,mesh):
        """
        Create subdomains by marking regions
        """
        subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
        subdomains.set_all(0) #assigns material/props number 0 everywhere
        return subdomains


    def create_Gauss_point_querry_domain(self,mesh):
        """
        Create subdomains by marking regions
        """
        GaussDomain = MeshFunction("size_t", mesh, mesh.topology().dim())
        GaussDomain.set_all(0) #assigns material/props number 0 everywhere
        GaussDomainQuerry= Gauss_point_Querry()
        GaussDomainQuerry.mark(GaussDomain,1)
        return GaussDomain

    
    def mark_boundaries(self,boundaries):
        """
        Mark left and right boundary points
        """
        boundaries.set_all(0)
        left0=left()
        left0.mark(boundaries,1)
        right0=right()
        right0.mark(boundaries,2)
        top0=top()
        top0.mark(boundaries,3)
        bottom0=bottom()
        bottom0.mark(boundaries,4)
        return
        
    def set_bcs(self):
        """
        Set boundary conditions for the user problem / could be replaced by external mesher, e.g. Abaqus, Gmsh...
        """
        if self.problem_step == 0:
            bcs=[
                [2,[2, [1],1.0]],
                # [1,[0, [0],0]],
                [1,[0, [1],0]],
                [4,[0, [0],0]],
                [3,[0, [0],0]]
                ]
        elif self.problem_step != 0:
            bcs=[
                [2,[2, [1],0.]],
                # [1,[0, [0],0]],
                [1,[0, [1],0]],
                [4,[0, [0],0.0]],
                [3,[0, [0],0.0]]
                ]
        return bcs    

    def history_output(self):
        """
        Used to get output of residual at selected node 
        """
        hist=[[2,[1,[1]]],
              [2,[0,[1]]],
              ]
        return hist

    def history_svars_output(self):
        """
        Used to get output of svars at selected Gauss point 
        """
        hist_svars=[[1,[1,[0]]], #sigma_11
                    [1,[1,[1]]], #sigma_22
                    [1,[1,[2]]], #sigma_33
                    [1,[1,[3]]], #sigma_23
                    [1,[1,[4]]], #sigma_13
                    [1,[1,[5]]], #sigma_12
                    [1,[1,[6]]], #epsilon_11
                    [1,[1,[7]]], #epsilon_22
                    [1,[1,[8]]], #epsilon_33
                    [1,[1,[9]]], #epsilon_23
                    [1,[1,[10]]], #epsilon_13
                    [1,[1,[11]]], #epsilon_12
                    [1,[1,[13]]], #epsilon_p_11
                    [1,[1,[14]]], #epsilon_p_22
                    [1,[1,[15]]], #epsilon_p_33
                    [1,[1,[16]]], #epsilon_p_23
                    [1,[1,[17]]], #epsilon_p_13
                    [1,[1,[18]]], #epsilon_p_12
                    [1,[1,[21]]], #lambda_dot
                    [1,[1,[22]]], #depsilon_p_11
                    [1,[1,[23]]], #depsilon_p_22
                    [1,[1,[24]]], #depsilon_p_33
                    [1,[1,[25]]], #depsilon_p_23
                    [1,[1,[26]]], #depsilon_p_13
                    [1,[1,[27]]]]  #depsilon_p_12
        return hist_svars    
    
    def set_materials(self):
        """
        Create material objects and set material parameters
        """
        mats=[]
        # load material #1
        
        env_lib=ngeo_parameters.env_lib        #umat_lib='./libplast_Cauchy3D-DP.so'
        umat_lib_path= ngeo_parameters.umat_lib_path
        umat_lib = umat_lib_path+'/CAUCHY3D-DP/libplast_Cauchy3D-DP.so'
        umat_id=2       # if many materials exist in the same library
        mat=UserMaterial(env_lib,umat_lib,umat_id)
        mat.props=self.set_material_1_properties(2000.,0.,100.)
        #
        mats.append(mat)
        return mats
    
    def set_material_1_properties(self,EE,nu,cc):
        """
        Sets material parameters
        """
        GG=EE/(2.*(1.+nu))
        KK=EE*GG/(3.*(3.*GG-EE))
        props=np.zeros(19)
        props[1-1]=KK
        props[2-1]=GG
        props[12-1]=cc
        props[15-1]=0.
        props[19-1]=1.
        props=props.astype("double")
        return props
    
    def give_me_solver_params(self,scale_t=1.):
            self.scale_t = scale_t
            self.slv.incmodulo = 1
            self.slv.dtmax=0.1*self.scale_t
            self.slv.tmax=1.*scale_t
            ninc=int(self.slv.tmax/self.slv.dtmax)   
            self.slv.nincmax=1000000
            self.slv.convergence_tol=10**-6
            self.slv.removezerolines=False
            print('hellooooo',self.scale_t, self.slv.tmax)
            
    def run_analysis_procedure(self,reference_data_path):
        saveto=reference_data_path+"./Cauchy_2D_Von_Mises_test_step_0.xdmf"
        self.problem_step = 0
        print(self.problem_step)
        self.bcs=self.set_bcs()
        self.feobj.symbolic_bcs = sorted(self.bcs, key=itemgetter(1))
        # print(cls.my_FEproblem.feobj.symbolic_bcs)
        print("initial")
        converged=self.solve(saveto,summary=True)
        scale_t_program = [10e5*self.scale_t,10e5*self.scale_t,10e5*self.scale_t,10e5*self.scale_t,10e5*self.scale_t]
        # print(scale_t_program)
        print("shearing1")
    
        nsteps=3
        for i in range(nsteps):
            print('step',i+1)
            self.problem_step = i+1
            scale_t = scale_t_program[i]
            # scale_t =10e5*self.scale_t
            self.slv.nincmax=1000000#1000000       
            self.slv.dtmax=0.001*scale_t
            self.slv.dt=self.slv.dtmax
            self.slv.tmax=self.slv.tmax+1.*scale_t
            self.feobj.symbolic_bcs = sorted(self.set_bcs(), key = itemgetter(1))
            self.feobj.initBCs()
            
            filename = 'Cauchy_2D_Von_Mises_test_step_'+str(i+1)
            saveto= reference_data_path+"./Cauchy_2D_Von_Mises_test_step_"+str(i+1)+".xdmf"
            converged=self.solve(saveto,summary=True)
        
        return converged
    
    def history_unpack(self,list1):
        for i,elem in enumerate(list1):
            if i==0:
                self.array_time=np.array([[elem[0]]])
                self.array_force=elem[1].reshape((1,len(elem[1])))
                self.array_disp=elem[2].reshape((1,len(elem[2])))
                continue
        
            self.array_time=np.concatenate((self.array_time.copy(),np.array([[elem[0]]])))
            self.array_force=np.concatenate((self.array_force.copy(),elem[1].reshape((1,len(elem[1])))))
            self.array_disp=np.concatenate((self.array_disp.copy(),elem[2].reshape((1,len(elem[2]))))) 

    def svars_history_unpack(self,list1):
        for i,elem in enumerate(list1):
            if i==0:
                self.array_dtime=np.array([[elem[0]]])
                self.array_gp_svars_comp=elem[1].reshape((1,len(elem[1])))
                # self.array_gp_2_svasr_comp=elem[2].reshape((1,2))

                # self.array_disp=elem[2].reshape((1,2))
                continue
            
            self.array_dtime=np.concatenate((self.array_dtime.copy(),np.array([[elem[0]]])))
            self.array_gp_svars_comp=np.concatenate((self.array_gp_svars_comp.copy(),elem[1].reshape((1,len(elem[1])))))
            # print(self.array_dtime)
            # print(self.array_gp_svars_comp.copy())
            # print(self.feobj.svars_history_indices)
            # self.array_disp=np.concatenate((self.array_gp_svasr_comp.copy(),elem[2].reshape((1,2)))) 

        
        # print(self.array_force)
        # return self.array_time, self.array_force, self.array_disp
    
    def extract_force_disp(self):
        analysis_history=self.feobj.problem_history
        # print(analysis_history)
        # analysis_history_flat=[elem1 for elem in analysis_history for elem1 in elem]
        self.history_unpack(analysis_history)
        self.array_time=self.array_time[:].copy()
        self.array_force=self.array_force[:].copy().sum(axis=1)/self.w
        self.array_force=self.array_force.reshape((-1,1))
        self.array_disp=self.array_disp[:,1].reshape((-1,1)).copy()

        
    def extract_elastoplastic_matrix(self):
        self.array_dforce=self.array_force[1:]-self.array_force[:-1]
        self.array_ddisp=self.array_disp[1:]-self.array_disp[:-1]
        # print(self.array_dforce)
        # print(self.array_ddisp)
        self.EH=np.divide(self.array_dforce[:],self.array_ddisp[:])
    
    def extract_svars_gauss_point(self):
        analysis_svars_history=self.feobj.problem_svars_history
        self.svars_history_unpack(analysis_svars_history)
        self.array_dtime=self.array_dtime[:].copy()
        self.array_gp_svars_comp=self.array_gp_svars_comp[:].copy()
# if __name__ == '__main__':
    
    