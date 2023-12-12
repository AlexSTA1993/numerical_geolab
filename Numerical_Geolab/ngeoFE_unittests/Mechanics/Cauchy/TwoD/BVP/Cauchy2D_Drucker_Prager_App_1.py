'''
Created on Nov 05, 2018

@author: Alexandros Stathas
'''
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
           
class Cauchy2DFEproblem(UserFEproblem):
    """
    Defines a user FE problem for given FE formulation
    """
    def __init__(self,FEformulation):
        self.description="Example of 2D plane strain problem, Cauchy continuum with Drucker Prager material"
        self.Normal_loading_eff=-200
        self.Normal_loading_total=-200
        self.problem_step=0
        self.h = 1.
        self.w = 1.#0.2
        
        self.ny=10 # element number along horizontal axes
        self.nw=10 # element number along vertical axes
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
        h=self.h
        w=self.w
        ny=self.ny#10#80
        nw=self.nw#10
        mesh = RectangleMesh(Point(-h/2.,-w/2.),Point(h/2.,w/2.),ny,nw,"left")#"crossed")
        #print(mesh.topology().dim())
        cd=MeshFunction("size_t", mesh, mesh.topology().dim())
        fd=MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        return mesh,cd,fd

    class left(SubDomain):
        def inside(self,x,on_boundary):
            return x[0] < -0.49  and on_boundary
    
    class right(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] > 0.49 and on_boundary
    
    class top(SubDomain):
        def inside(self,x,on_boundary):
            return x[1] > 0.49  and on_boundary
    
    class bottom(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] < -0.49 and on_boundary
    
    class Gauss_point_Querry(SubDomain):
        def __init__(self,w,nw,h,ny):
            self.w=w
            self.nw=nw
            self.h=h
            self.ny=ny
            super().__init__()
            
        def inside(self, x, on_boundary):
            # return x[0] >= 1./2.-1./80. and between(x[1], (-0.1,0.1))
            return between(x[0], (-self.w/(self.nw),self.w/(self.nw))) and between(x[1], (-self.h/(self.ny),self.h/(self.ny)))


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
        GaussDomainQuerry= self.Gauss_point_Querry(self.w,self.nw,self.h,self.ny)
        GaussDomainQuerry.mark(GaussDomain,1)
        return GaussDomain
    
    def mark_boundaries(self,boundaries):
        """
        Mark left and right boundary points
        """
        boundaries.set_all(0)
        left0=self.left()
        left0.mark(boundaries,1)
        right0=self.right()
        right0.mark(boundaries,2)
        top0=self.top()
        top0.mark(boundaries,3)
        bottom0=self.bottom()
        bottom0.mark(boundaries,4)
        return
    
    def set_initial_conditions(self):
        """
        Initialize state variables vector
        """
        #Modify the state variables (corresponding to the stresses)
        print(self.genprops.p_nsvars)
        tmp=np.zeros(self.genprops.p_nsvars)
        tmp[1-1]=self.Normal_loading_eff
        tmp[2-1]=self.Normal_loading_eff
        tmp[3-1]=self.Normal_loading_eff
        # tmp[53-1]= self.Pressure_loading
    
        self.feobj.svars2.interpolate(Constant(tmp))
    
        #Modify the stresses (for Paraview)
        tmp=np.zeros(3)
        # tmp=np.zeros(6)
        tmp[1-1]=self.Normal_loading_total #specimen is in dry conditions total pressure equals effective pressure
        tmp[2-1]=self.Normal_loading_total
        # tmp[3-1]=self.Normal_loading_total
    
        self.feobj.sigma2.interpolate(Constant(tmp))

        tmp=np.zeros(2)        
        self.feobj.usol.interpolate(Constant(tmp))
    
        return
        
    def set_bcs(self):
        """
        Set boundary conditions for the user problem / could be replaced by external mesher, e.g. Abaqus, Gmsh...
        """
        scale_d =1.
        u1=-(0.1+0.2*0.5)/(np.sqrt(2./3.*3.5)-0.25*2/3)#
        # u1=0.02449428 #We choose this value to be as close to the real yield limit. Then yield starts at the new step and ldot is calculated by the material algorithm and not the external Newton loop
        if self.problem_step == 0:
            bcs=[
                [2,[0,[0],0.]],#self.Normal_loading_total
                [1,[2,[0],0.]],
                [3,[0,[1],0.]],
                [4,[2,[1],0.]],
                ]
        elif self.problem_step != 0:
            bcs=[
                # [2,[3, [0], self.Normal_loading_total]],#self.Normal_loading_total
                [2,[0, [0], u1]],#self.Normal_loading_total
                [1,[0, [0], 0.]],
                [4,[0, [1],0]],
                [3,[0, [1], -0.5*u1]],
                ]
        return bcs    

    def history_output(self):
        """
        Used to get output of residual at selected node 
        """
        hist=[[2,[1,[0]]],
              [2,[0,[0]]],
              [2,[1,[1]]],
              [2,[0,[1]]],
              [3,[1,[0]]],
              [3,[0,[0]]],
              [3,[1,[1]]],
              [3,[0,[1]]],
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
                    [1,[1,[21]]], #lambda_dod
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
        mat.props=self.set_material_1_properties(2000.,0.,0.5,100.,0.)
        #
        mats.append(mat)
        return mats
    
    def set_material_1_properties(self,EE,nu,tanfi,cc,tanpsi):
        """
        Sets material parameters
        """
        GG=EE/(2.*(1.+nu))
        KK=EE*GG/(3.*(3.*GG-EE))
        props=np.zeros(19)
        props[1-1]=KK
        props[2-1]=GG
        props[11-1]=tanfi
        props[12-1]=cc
        props[13-1]=tanpsi
        props[15-1]=-0.1
        props[19-1]=0.
        props=props.astype("double")
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
        print('hello!!!')
        saveto=reference_data_path+"Cauchy_2D_Drucker-Prager_test_step_0"+"_App_1"+".xdmf"
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
#         print('step',i)
            self.problem_step = i+1
            scale_t = scale_t_program[i]
            self.slv.nincmax=ninc[i]#1000000       
            self.slv.dtmax=0.1*scale_t
            self.slv.dt=self.slv.dtmax
            self.slv.tmax=self.slv.tmax+1.*scale_t
            self.feobj.symbolic_bcs = sorted(self.set_bcs(), key = itemgetter(1))
            self.feobj.initBCs()
#         print('tmax', my_FEproblem.slv.tmax)
            filename = 'Cauchy_2D_Drucker-Prager_test_step_'+str(i+1)
            saveto= reference_data_path+"Cauchy_2D_Drucker-Prager_test_step_"+str(i+1)+"_App_1"+".xdmf"
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
    
    