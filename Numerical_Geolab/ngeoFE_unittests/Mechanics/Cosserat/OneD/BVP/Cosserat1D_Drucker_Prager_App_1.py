'''
Created on Nov 05, 2018

@author: Alexandros Stathas
'''
import os

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

import os
from _operator import itemgetter
# from tkinter.constants import NW

warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

class Cosserat1DFEformulation(FEformulation):
    '''
    Defines a user FE formulation
    '''
    def __init__(self):
        # Number of stress/deformation components
        self.p_nstr=4
        # Number of Gauss points
        self.ns=1
    
    def generalized_epsilon(self,v):
        """
        Set user's generalized deformation vector
        """
        scale_u=1.#1./1000.
        gde=[
            Dx(v[0],0)*scale_u,#gamma_11
            v[2]*scale_u,                    #gamma_12
            Dx(v[1],0)*scale_u-v[2]*scale_u,         #gamma_21
            Dx(v[2],0)*scale_u,              #kappa_31
            ]
        return as_vector(gde)
    
    def create_element(self,cell):
        """
        Set desired element
        """
        self.degree=1
        element1=VectorElement("Lagrange",cell,degree=self.degree,dim=2)
        element2=FiniteElement("Lagrange",cell,degree=self.degree)

        element=MixedElement([element1,element2])
        return element

           
class Cosserat1DFEproblem(UserFEproblem):
    """
    Defines a user FE problem for given FE formulation
    """
    def __init__(self,FEformulation):
        self.description="Example of 1D problem, Cosserat continuum with Drucker Prager material"
        scale = 1.
        self.problem_step=0
        self.Pressure_loading = 0.*200./3.*scale
        self.Normal_loading_eff = -600./3*scale+self.Pressure_loading
        self.Normal_loading_total =self.Normal_loading_eff-self.Pressure_loading
        super().__init__(FEformulation)

    
    def set_general_properties(self):
        """
        Set here all the parameters of the problem, except material properties 
        """
        self.genprops=General_FEproblem_properties()
        # Number of state variables
        self.genprops.p_nsvars=85
    
    def create_mesh(self):
        """
        Set mesh and subdomains
        """
        self.w=1.
        self.nw=int(1)
        mesh=IntervalMesh(self.nw,-self.w/2.,self.w/2.)
        cd = MeshFunction("size_t", mesh, mesh.topology().dim())
        fd = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        return mesh, cd, fd

    def create_subdomains(self,mesh):
        """
        Create subdomains by marking regions
        """
        subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
        subdomains.set_all(0) #assigns material/props number 0 everywhere
#         imperfection = self.Imperfection(self.n,self.h)
#         imperfection.mark(subdomains,1)
        return subdomains
       
    class Boundary(SubDomain):
        def __init__(self,xyz,param):
            self.xyz=xyz
            self.param=param
            super().__init__()
        def inside(self, x, on_boundary):
            tol = DOLFIN_EPS
            return on_boundary and near(x[self.xyz],self.param)    
            
    class Gauss_point_Querry(SubDomain):
        def __init__(self,w,nw):
            self.w=w
            self.nw=nw
            super().__init__()
            
        def inside(self, x, on_boundary):
            # return x[0] >= 1./2.-1./80. and between(x[1], (-0.1,0.1))
            return between(x[0], (-self.w/(self.nw),self.w/(self.nw)))

    def create_Gauss_point_querry_domain(self,mesh):
        """
        Create subdomains by marking regions
        """
        GaussDomain = MeshFunction("size_t", mesh, mesh.topology().dim())
        GaussDomain.set_all(0) #assigns material/props number 0 everywhere
        GaussDomainQuerry= self.Gauss_point_Querry(self.w,self.nw)
        GaussDomainQuerry.mark(GaussDomain,1)
        return GaussDomain

    
    def mark_boundaries(self, boundaries):
        """
        Mark left and right boundary points
        """
        left0 = self.Boundary(0,-self.w/2.)
        left0.mark(boundaries, 1)
        right0 = self.Boundary(0,self.w/2.)
        right0.mark(boundaries, 2)
        #         
        return
  
    def set_initial_conditions(self):
        """
        Initialize state variables vector
        """
        #Modify the state variables (corresponding to the stresses)
        tmp=np.zeros(self.genprops.p_nsvars)
        tmp[1-1]=self.Normal_loading_eff#+self.Pressure_loading 
        tmp[5-1]=self.Normal_loading_eff#*5./8.
        tmp[9-1]=self.Normal_loading_eff#*5./8.

        self.feobj.svars2.interpolate(Constant(tmp))
        
        #Modify the stresses (for Paraview)
        tmp=np.zeros(4)
        tmp[1-1]=self.Normal_loading_total
        self.feobj.sigma2.interpolate(Constant(tmp))
        tmp=np.zeros(3)
        self.feobj.usol.interpolate(Constant(tmp))
                    
        pass
        
    def set_bcs(self):
        """
        Set boundary conditions for the user problem / could be replaced by external mesher, e.g. Abaqus, Gmsh...
        """

        p = self.Normal_loading_eff
        tanfi = self.mats[0].props[10]
        G = self.mats[0].props[1]
        Gc = self.mats[0].props[2]
        h1 = self.mats[0].props[15]
        h2 = self.mats[0].props[16]
        # self.u1_tot=p*tanfi/(np.sqrt(2*(h1+h2)*G**2.+2*(h1-h2)*Gc**2.))
        self.u1_tot=p*tanfi/(np.sqrt(2*(h1+h2))*(G-Gc))

        scale_u=1.#1./1000.

        bcs=[]
        if self.problem_step == 0:
            bcs = [
            
                [1, [0, [0,0], 0.]],
                [1, [0, [0,1], 0.]],
                [1, [1, [1], 0.]],                
                
                [2, [0, [0,0], 0]],
                [2, [0, [0,1], 0]],
                # [2, [3, [0,0], self.Normal_loading_total*scale_u]],
                [2, [1, [1], 0.]],                
                ]
        elif self.problem_step == 1:
            bcs = [
            
                [1, [0, [0,0], 0.]],
                [1, [0, [0,1], 0.]],
                [1, [1, [1], 0.]],
                [2, [0, [0,0], 0.]],
                # [2, [3, [0,0], self.Normal_loading_total*scale_u]],
                [2, [0, [0,1], self.u1_tot/scale_u]],
                [2, [1, [1], 0.]],                   
                ]
        elif self.problem_step > 1:
            bcs = [
            
                [1, [0, [0,0], 0.]],
                [1, [0, [0,1], 0.]],
                [1, [1, [1], 0.]],
                [2, [0, [0,0], 0.]],
                # [2, [3, [0,0], self.Normal_loading_total*scale_u]],
                [2, [0, [0,1], self.u1_tot/scale_u]],
                [2, [1, [1], 0.]], 
                ]           
        return bcs


    def history_output(self):
        """
        Used to get output of residual at selected node 
        """
        hist=[[1,[1,[0,0]]],
              [1,[0,[0,0]]],
              [1,[1,[0,1]]],
              [1,[0,[0,1]]],
              [1,[0,[1]]],
              [1,[1,[1]]],

              [2,[1,[0,0]]],
              [2,[0,[0,0]]],
              [2,[1,[0,1]]],
              [2,[0,[0,1]]],
              [2,[0,[1]]],
              [2,[1,[1]]],
              ]
        return hist

    def history_svars_output(self):
        """
        Used to get output of svars at selected Gauss point 
        """
        hist_svars=[[1,[1,[0]]], #tau_11
                    [1,[1,[1]]], #tau_12
                    [1,[1,[2]]], #tau_13
                    [1,[1,[3]]], #tau_21
                    [1,[1,[4]]], #tau_22
                    [1,[1,[5]]], #tau_23
                    [1,[1,[6]]], #tau_31
                    [1,[1,[7]]], #tau_32
                    [1,[1,[8]]], #tau_33
                    [1,[1,[9]]], #mu_11
                    [1,[1,[10]]], #mu_12
                    [1,[1,[11]]], #mu_13
                    [1,[1,[12]]], #mu_21
                    [1,[1,[13]]], #mu_22
                    [1,[1,[14]]], #mu_23
                    [1,[1,[15]]], #mu_31
                    [1,[1,[16]]], #mu_32
                    [1,[1,[17]]], #mu_33
                    [1,[1,[18]]], #gamma_11
                    [1,[1,[19]]], #egamma_12
                    [1,[1,[20]]], #gamma_13
                    [1,[1,[21]]], #gamma_21
                    [1,[1,[22]]], #gamma_22
                    [1,[1,[23]]], #gamma_23
                    [1,[1,[24]]], #gamma_31
                    [1,[1,[25]]], #gamma_32
                    [1,[1,[26]]], #gamma_33
                    [1,[1,[27]]], #kappa_11
                    [1,[1,[28]]], #kappa_12
                    [1,[1,[29]]], #kappa_13
                    [1,[1,[30]]], #kappa_21
                    [1,[1,[31]]], #kappa_22
                    [1,[1,[32]]], #kappa_23
                    [1,[1,[33]]], #kappa_31
                    [1,[1,[34]]], #kappa_32
                    [1,[1,[35]]], #kappa_33
                    [1,[1,[57]]], #lambda_dot
                    ]
        return hist_svars    
    
    def set_materials(self):
        """
        Create material objects and set material parameters
        """
        mats=[]
        # load material #1
        
        env_lib=ngeo_parameters.env_lib       
        umat_lib_path= ngeo_parameters.umat_lib_path
        umat_lib = umat_lib_path+'/COSSERAT3D/libplast_Cosserat3D.so'
        umat_id=1      # if many materials exist in the same library
        mat=UserMaterial(env_lib,umat_lib,umat_id)
        mat.props=self.set_material_1_properties()
        #
        mats.append(mat)
        return mats
    
    def set_material_1_properties(self):
        """
        Sets material parameters
        """
        
        g1=8./5.;g2=2./5.;g3=0 ;g4=0.

        h1=2./3. ;h2=-1./6.;h3=2./3.;h4=-1./6.;
        
        K=666.66; G=1.*10.**3.; Gc=0.5*10.**3. ; L=1*10**(2.);R=10.*10.**(-3.);  
        MG=G*(R**2.)/h3 ; MGc=MG;  
        tanfi=0.5; cc=0.;
        tanpsi=0.; Hsfi=0.; Hscc=-0.;
        eta1=0.0
        
        prop_num=29
        props=np.zeros(prop_num)
        props[0]=K
        props[1]=G
        props[2]=Gc
        props[3]=L
        props[4]=MG
        props[5]=MGc
        props[9]=R
        props[10]=tanfi
        props[11]=cc
        props[12]=tanpsi
        props[13]=Hsfi
        props[14]=Hscc
        props[15]=h1
        props[16]=h2
        props[17]=h3
        props[18]=h4
        props[19]=g1
        props[20]=g2
        props[21]=g3
        props[22]=g4
        props[23]=eta1
        
        return props

    
    def give_me_solver_params(self,scale_t=1.):
        self.scale_t = scale_t
        self.slv.incmodulo = 1
        self.slv.dtmax=1.0*self.scale_t
        self.slv.tmax=1.*scale_t
        ninc=int(self.slv.tmax/self.slv.dtmax)   
        self.slv.nincmax=50
        self.slv.convergence_tol=10**-6
        self.slv.removezerolines=False
            
    def run_analysis_procedure(self,reference_data_path):
        saveto=reference_data_path+"Cosserat_1D_Drucker-Prager_test_step_0"+"_App_1"+".xdmf"
        self.problem_step = 0
        self.bcs=self.set_bcs()
        self.feobj.symbolic_bcs = sorted(self.bcs, key=itemgetter(1))
        print("initial")
        converged=self.solve(saveto,summary=True)
        scale_t_program = [self.scale_t,self.scale_t,self.scale_t,self.scale_t,self.scale_t,self.scale_t]
        ninc=[100,100,100,100,100,100]
        print("shearing1")
    
        nsteps=6
        for i in range(nsteps):
            self.problem_step = i+1
            scale_t = scale_t_program[i]
            self.slv.nincmax=ninc[i] 
            self.slv.dtmax=0.1*scale_t
            self.slv.dt=self.slv.dtmax
            self.slv.tmax=self.slv.tmax+1.*scale_t
            self.feobj.symbolic_bcs = sorted(self.set_bcs(), key = itemgetter(1))
            self.feobj.initBCs()
            filename = 'Cosserat_1D_Drucker-Prager_test_step_'+str(i+1)
            saveto= reference_data_path+"Cosserat_1D_Drucker-Prager_test_step_"+str(i+1)+"_App_1"+".xdmf"
            converged=self.solve(saveto,summary=True)
        
        return converged
    
    def history_unpack(self,list1):
        for i,elem in enumerate(list1):
            # print(elem)
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
                # print(elem)
                self.array_dtime=np.array([[elem[0]]])
                self.array_gp_svars_comp=elem[1].reshape((1,len(elem[1])))
                continue
            
            self.array_dtime=np.concatenate((self.array_dtime.copy(),np.array([[elem[0]]])))
            self.array_gp_svars_comp=np.concatenate((self.array_gp_svars_comp.copy(),elem[1].reshape((1,len(elem[1])))))
    
    def extract_force_disp(self):
        analysis_history=self.feobj.problem_history
        self.history_unpack(analysis_history)
        self.array_time=self.array_time[:].copy()
        self.array_force=self.array_force[:,:]#.reshape((-1,20))
        self.array_disp=self.array_disp[:,:]#.reshape((-1,20)).copy()
        return
    

    def extract_svars_gauss_point(self):
        analysis_svars_history=self.feobj.problem_svars_history
        self.svars_history_unpack(analysis_svars_history)
        self.array_dtime=self.array_dtime[:].copy()
        self.array_gp_svars_comp=self.array_gp_svars_comp[:,:].copy()
# if __name__ == '__main__':
    
    