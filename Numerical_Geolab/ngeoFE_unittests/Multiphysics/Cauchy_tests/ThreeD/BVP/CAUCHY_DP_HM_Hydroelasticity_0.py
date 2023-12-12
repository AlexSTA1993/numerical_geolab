'''
Created on Jul 29, 2019

@author: Alexandros STATHAS

BVP Hydroelasticity, sets initial conditions and pressure at the bottom of an insulated 3D cantilever.
'''

import os


from dolfin import *
import pickle
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec

#
from ngeoFE.feproblem import UserFEproblem, General_FEproblem_properties
from ngeoFE.fedefinitions import FEformulation
from ngeoFE.materials import UserMaterial
#
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from dolfin.cpp.io import HDF5File
from numpy.core.tests.test_getlimits import assert_ma_equal
from _operator import itemgetter
#from dolfin.cpp.mesh import MeshFunction

warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
#from Parametric_Cosserat import Cosserat_1D_FEformulation

from ngeoFE_unittests import ngeo_parameters


class THM3D_FEformulation(FEformulation):
    '''
    Defines a user FE formulation
    '''
    def __init__(self):
        # Number of stress/deformation components
        self.p_nstr=6+3+3
        # Number of Gauss points
        self.ns=1
        # Number of auxiliary quantities at Gauss points
        self.p_aux=2
        
    def generalized_epsilon(self,v):
        '''
        Set user's generalized deformation vector
        '''
        gde=[
            Dx(v[0],0),              #gamma_11
            Dx(v[1],1),              #gamma_22
            Dx(v[2],2),              #gamma_22
            Dx(v[1],2)+Dx(v[2],1),   #gamma_23
            Dx(v[0],2)+Dx(v[2],0),   #gamma_13
            Dx(v[0],1)+Dx(v[1],0),   #gamma_12
            Dx(v[3],0),  #q_1 - pf
            Dx(v[3],1),  #q_2 - pf
            Dx(v[3],2),  #q_3 - pf
            Dx(v[4],0),  #q_1 - temp
            Dx(v[4],1),  #q_2 - temp
            Dx(v[4],2),  #q_3 - temp
            ]
        return as_vector(gde)
    
    def auxiliary_fields(self,v):
        '''
        Set user's generalized deformation vector
        '''
        auxgde=[
        v[3],
        v[4],
            ]
        return as_vector(auxgde)
    
    def setVarFormAdditionalTerms_Res(self,u,Du,v,svars,metadata,dt):
        Res=0.
        lstar=svars.sub(55-1)
        bstar=svars.sub(56-1)
        rhoC=svars.sub(57-1)
        #HM terms
        eps=self.generalized_epsilon(Du)
        eps_v=eps[0]+eps[1]+eps[2]
        virtual_pf=v[3]
        
        Res+=-(1./dt)*(1./bstar)*dot(eps_v,virtual_pf)*dx(metadata=metadata) 
        
        #TM terms
        virtual_Temp=v[4]
        for i in range(1,6):
            Res+= +(1./dt)* (1./rhoC)*svars.sub(1+i-1)*svars.sub(41+i-1)*virtual_Temp*dx(metadata=metadata)
          
        #HT terms
        DTemp=Du[4]
        Res+= +(1./dt)*(lstar/bstar)*dot(DTemp,virtual_pf)*dx(metadata=metadata)
        
        return Res
    
    def setVarFormAdditionalTerms_Jac(self,u,Du,v,svars,metadata,dt,ddsdde):
        lstar=svars.sub(55-1)
        bstar=svars.sub(56-1)
        rhoC=svars.sub(57-1)
        alfa=svars.sub(58-1)
        Jac=0.
        #HM terms
        eps=self.generalized_epsilon(u) #needs u (trial function, because it takes derivatives in terms of u and not Du for calculating the Jacobian.
        eps_vol=eps[0]+eps[1]+eps[2]
        virtual_pf=v[3]
        Jac+=+(1./dt)*(1./bstar)*dot(eps_vol,virtual_pf)*dx(metadata=metadata)
 
        #MH terms
        pf=u[3] #same as before
        virtual_eps=self.generalized_epsilon(v)
        virtual_eps_vol=virtual_eps[0]+virtual_eps[1]+virtual_eps[2]
        Jac+=-(1./dt)*dt*dot(pf,virtual_eps_vol)*dx(metadata=metadata)
         
        #HT terms
        temperature = u[4]
        Jac+=-(1./dt)*(lstar/bstar)*dot(temperature,virtual_pf)*dx(metadata=metadata)
        
        #TH terms
        eps_temp=alfa*temperature*as_vector([1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        Jac+=-(1./dt)*dt*inner(dot(ddsdde,eps_temp),virtual_eps)*dx(metadata=metadata) 
        
        #TM terms due to thermal expansion and plastic deformation
        virtual_temp=v[4]
        eps_eff=eps+eps_temp
        eps_plastic=[]
        for i in range(0,self.p_nstr):
            eps_plastic.append(svars.sub(41-1+i))
        eps_plastic=as_vector(eps_plastic)
        Jac+=-(1./dt)*dt*(1./rhoC)*inner(dot(ddsdde,eps_eff),eps_plastic)*virtual_temp*dx(metadata=metadata)
        #TM terms due to fluid pressure and plastic deforamtion, i.e. thermal pressurization 
        eps_plastic_vol=eps_plastic[0]+eps_plastic[1]+eps_plastic[2]
        Jac+=+(1./dt)*dt*(1./rhoC)*pf*eps_plastic_vol*virtual_temp*dx(metadata=metadata)
         
        #TM 2
        Jac+=dt*(1./rhoC)*inner(dot(ddsdde,eps_temp),eps_plastic)*virtual_temp*dx(metadata=metadata)
        return Jac
    
    def create_element(self,cell):
        """
        Set desired element
        """
        self.degree=1
        # Defines a Lagrangian FE of degree 1 for the displacements
        element=VectorElement("Lagrange",cell,degree=self.degree,dim=3+1+1)

        return element

    def dotv_coeffs(self):
        """   
        Set left hand side derivative coefficients
        """
        return as_vector([0.,0.,0.,1.,1.])
    
class THM3D_FEproblem(UserFEproblem):
    def __init__(self,FEformulation):
        self.description="Example of 3D plane strain problem, Cauchy continuum"
        self.problem_step=0
        self.h = 1.
        self.Normal_loading_total=-200.
        self.Pressure_loading=66.66
        self.Normal_loading_eff=self.Normal_loading_total+self.Pressure_loading
        super().__init__(FEformulation)
    
    def set_general_properties(self):
        """
        Set here all the parameters of the problem, except material properties 
        """
        self.genprops=General_FEproblem_properties()
        # Number of state variables
        self.genprops.p_nsvars=62
        
    def create_mesh(self):
        """
        Set mesh and subdomains
        """
        self.h1=1
        self.h2=1
        self.h3=10.
        self.nx=1
        self.ny=1
        self.nz=100
        mesh=BoxMesh(Point(-0.5*self.h1,-0.5*self.h2,-0.5*self.h3),Point(0.5*self.h1,0.5*self.h2,0.5*self.h3),self.nx,self.ny,self.nz)        
      
        cd = MeshFunction("size_t", mesh, mesh.topology().dim())
        fd = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        return mesh, cd, fd
    
    def create_subdomains(self,mesh):
        """
        Create subdomains by marking regions
        """
        subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
        subdomains.set_all(0) #assigns material/props number 0 everywhere
        return subdomains
    
    class Boundary(SubDomain):
        def __init__(self,xyz,param):
            self.xyz=xyz
            self.param=param
            super().__init__()
        def inside(self, x, on_boundary):
            tol = DOLFIN_EPS
            return on_boundary and near(x[self.xyz],self.param)    
        
    class Cornerpoint(SubDomain):
        def __init__(self,*argz):
            self.xyz=[]
            self.param=[]
            for arg in argz:
                self.xyz.append(arg[0])
                self.param.append(arg[1])

            super().__init__()
        def inside(self, x, on_boundary):
            tol = DOLFIN_EPS
            return near(x[self.xyz[0]],self.param[0],tol) and near(x[self.xyz][1],self.param[1],tol) and near(x[self.xyz[2]],self.param[2],tol)
    
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
        
        corner_point1 = self.Cornerpoint([2,-self.h3/2.],[0,-self.h1/2.],[1,-self.h2/2.])
        corner_point1.mark(boundaries, 7)
        
        corner_point2 = self.Cornerpoint([2,-self.h3/2.],[0,self.h1/2.],[1,self.h2/2.])
        corner_point2.mark(boundaries, 8)
        
        corner_point3 = self.Cornerpoint([2,-self.h3/2.],[0,self.h1/2.],[1,-self.h2/2.])
        corner_point3.mark(boundaries, 9)
        
        corner_point4 = self.Cornerpoint([2,-self.h3/2.],[0,-self.h1/2.],[1,self.h2/2.])
        corner_point4.mark(boundaries, 10)

        return

    def set_initial_conditions(self):
        """
        Initialize state variables vector
        """
        #Modify the state variables (corresponding to the stresses)

        tmp=np.zeros(self.genprops.p_nsvars)
        tmp[1-1]=self.Normal_loading_eff
        tmp[2-1]=self.Normal_loading_eff
        tmp[3-1]=self.Normal_loading_eff
        tmp[53-1]= self.Pressure_loading
    
        self.feobj.svars2.interpolate(Constant(tmp))
    
        #Modify the stresses (for Paraview)
        tmp=np.zeros(12)

        tmp[1-1]=self.Normal_loading_total #specimen is in dry conditions total pressure equals effective pressure
        tmp[2-1]=self.Normal_loading_total
        tmp[3-1]=self.Normal_loading_total
    
        self.feobj.sigma2.interpolate(Constant(tmp))
    
        tmp=np.zeros(5)
        tmp[4-1]= self.Pressure_loading 
        self.feobj.usol.interpolate(Constant(tmp))
    
        return
  
    def set_bcs(self):
        """
        Set boundary conditions for the user problem / could be replaced by external mesher, e.g. Abaqus, Gmsh...
        """
        self.DP=10.
        self.DT=0.

        if self.problem_step == 0:
            bcs=[
                [1,[3,[2],self.Normal_loading_total]],
                [4,[3,[0],self.Normal_loading_total]],
                [6,[3,[1],self.Normal_loading_total]],
                [2,[0,[2],0.]],
                [3,[0,[0],0.]],
                [5,[0,[1],0.]],
                ]
        elif self.problem_step != 0:
            bcs=[
                [1,[3,[2],self.Normal_loading_total]],
                [2,[0,[2],0.]],
                [3,[0,[0],0.]],
                [4,[3,[0],self.Normal_loading_total]],
                [5,[0,[1],0.]],
                [6,[3,[1],self.Normal_loading_total]],
                
                [1,[3,[3],0.]],
                [2,[2,[3],self.DP]],
                [1,[3,[4],0.]],
                [2,[2,[4],self.DT]],
               
                ]
        return bcs    
    
    def history_output(self):
        """
        Used to get output of residual at selected node 
        """
        hist=[[1,[1,[0]]],
              [1,[0,[0]]],
              [1,[1,[1]]],
              [1,[0,[1]]],
              [1,[1,[2]]],
              [1,[0,[2]]],
              [1,[1,[3]]],
              [1,[0,[3]]],
              [1,[1,[4]]],
              [1,[0,[4]]],
              ]
        return hist
    
    def set_materials(self):
        """
        Create material objects and set material parameters
        """
        mats=[]
        # load material #1
        env_lib=ngeo_parameters.env_lib 
        umat_lib=ngeo_parameters.umat_lib_path+'CAUCHY3D-DP-PR-TEMP/libplast_Cauchy3D-DP-PR-TEMP.so'
        
        umat_id=3 # if many materials exist in the same library
        mat=UserMaterial(env_lib,umat_lib,umat_id)
        mat.props=self.set_material_1_properties()
        mats.append(mat)
        return mats
    
    def set_material_1_properties(self):
        """
        Sets material parameters
        """
        EE=2000.;nu=0. ;cc=10000000. ;tanfi=0.;tanpsi=0.;Hsfi=0.;Hscc=0.
        GG=EE/(2.*(1.+nu))
        KK=EE*GG/(3.*(3.*GG-EE))
        permeability=1.;fluid_viscosity=1.;lstar=10.**-8;bstar=10.**16;alpha=0.
        conductivity=1./1.;rhoC=1. #density * specific heat capacity
        props=np.array([KK,GG,permeability,fluid_viscosity,bstar,conductivity,rhoC,alpha,lstar,0.,tanfi,cc,tanpsi,Hsfi,Hscc,0.,0.,0.,0.,0.])
        props=props.astype("double")
        return props

    def give_me_solver_params(self,scale_t=1.):
            self.scale_t = scale_t
            self.slv.incmodulo = 1
            self.slv.dtmax=0.1
            self.slv.tmax=1.0
            ninc=int(self.slv.tmax/self.slv.dtmax)   
            self.slv.nincmax=1000000
            self.slv.convergence_tol=10**-8
            self.slv.removezerolines=False
            
    def run_analysis_procedure(self,reference_data_path):
        saveto=reference_data_path+"THM-RESULTS/HYDRO_ELASTIC/Hydroelasticity_test3D_THM3D_step_0.xdmf"
        self.problem_step = 0
        self.bcs=self.set_bcs()
        self.feobj.symbolic_bcs = sorted(self.bcs, key=itemgetter(1))
        print("initial")
        converged=self.solve(saveto,summary=True)
        scale_t_program = [self.scale_t,self.scale_t,self.scale_t,self.scale_t,self.scale_t]
        ninc=[5010,100,100,100]
        print("shearing1")
    
        nsteps=1
        for i in range(nsteps):
            self.problem_step = i+1
            scale_t = scale_t_program[i]
            self.slv.nincmax=ninc[i]
            self.slv.dtmax=5.*scale_t
            self.slv.dt=self.slv.dtmax
            self.slv.tmax=self.slv.tmax+500.*scale_t
            self.feobj.symbolic_bcs = sorted(self.set_bcs(), key = itemgetter(1))
            self.feobj.initBCs()

            filename = 'THM-RESULTS/HYDRO_ELASTIC/Hydroelasticity_test3D_THM3D_step_'+str(i+1)
            saveto= reference_data_path+filename+".xdmf"
            converged=self.solve(saveto,summary=True)
        
        return converged
    
    def history_unpack(self,list1):
        for i,elem in enumerate(list1):
            if i==0:
                self.array_time=np.array([[elem[0]]])
                self.array_gen_force=elem[1].reshape((1,len(elem[1])))
                self.array_gen_disp=elem[2].reshape((1,len(elem[2])))
            
                continue
        
            self.array_time=np.concatenate((self.array_time.copy(),np.array([[elem[0]]])))
            self.array_gen_force=np.concatenate((self.array_gen_force.copy(),elem[1].reshape((1,len(elem[1])))))
            self.array_gen_disp=np.concatenate((self.array_gen_disp.copy(),elem[2].reshape((1,len(elem[2]))))) 
        return 
    
    def extract_generalized_force_disp(self):
        analysis_history=self.feobj.problem_history
        self.history_unpack(analysis_history)
        self.array_time=self.array_time[:].copy()
        self.array_gen_force=self.array_gen_force[:,:]
        self.array_gen_disp=self.array_gen_disp[:,:]
        return
        
    def extract_elastoplastic_matrix(self):
        self.array_gen_dforce=self.array_gen_force[1:]-self.array_gen_force[:-1]
        self.array_gen_ddisp=self.array_gen_disp[1:]-self.array_gen_disp[:-1]
        self.EH=np.divide(self.array_gen_dforce[:],self.array_gen_ddisp[:])
        return
