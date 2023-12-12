'''
Created on Dec 30, 2022

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

class Cosserat3DFEformulation(FEformulation):
    '''
    Defines a user FE formulation
    '''
    def __init__(self):
        # Number of stress/deformation components
        self.p_nstr=18
        # Number of Gauss points
        self.ns=1
    
    def generalized_epsilon(self,v):
        """
        Set user's generalized deformation vector
        """
        scale_u=1.#1./1000.
        gde=[
            Dx(v[0],0)*scale_u,              #gamma_11
            Dx(v[0],1)*scale_u+v[5]*scale_u, #gamma_12
            Dx(v[0],2)*scale_u-v[4]*scale_u, #gamma_13
            Dx(v[1],0)*scale_u-v[5]*scale_u, #gamma_21

            Dx(v[1],1)*scale_u,              #gamma_22
            Dx(v[1],2)*scale_u+v[3]*scale_u, #gamma_23
            Dx(v[2],0)*scale_u+v[4]*scale_u, #gamma_31
            Dx(v[2],1)*scale_u-v[3]*scale_u, #gamma_32
            Dx(v[2],2)*scale_u,              #gamma_33
            
            Dx(v[3],0)*scale_u,              #kappa_11
            Dx(v[3],1)*scale_u,              #kappa_12
            Dx(v[3],2)*scale_u,              #kappa_13
            Dx(v[4],0)*scale_u,              #kappa_21
            Dx(v[4],1)*scale_u,              #kappa_22
            Dx(v[4],2)*scale_u,              #kappa_23
            Dx(v[5],0)*scale_u,              #kappa_31
            Dx(v[5],1)*scale_u,              #kappa_32
            Dx(v[5],2)*scale_u,              #kappa_33
            ]
        return as_vector(gde)
    
    def create_element(self,cell):
        """
        Set desired element
        """
        self.degree=1
        element1=VectorElement("Lagrange",cell,degree=self.degree,dim=3)
        element2=VectorElement("Lagrange",cell,degree=self.degree,dim=3)

        element=MixedElement([element1,element2])
        return element

           
class Cosserat3DFEproblem(UserFEproblem):
    """
    Defines a user FE problem for given FE formulation
    """
    def __init__(self,FEformulation):
        self.description="Example of 2D plane strain problem, Cosserat continuum with Drucker Prager material"
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
        self.h1=1.
        self.h2=1.
        self.h3=1.
        self.nx=1
        self.ny=1
        self.nz=1
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
        def __init__(self,h1,h2,h3,nx,ny,nz):
            self.h1=h1
            self.h2=h2
            self.h3=h3
            self.nx=nx
            self.ny=ny
            self.nz=nz
            super().__init__()
            
        def inside(self, x, on_boundary):
            # return x[0] >= 1./2.-1./80. and between(x[1], (-0.1,0.1))
            return between(x[0], (-self.h1/(self.nx),self.h1/(self.nx))) and between(x[1], (-self.h2/(self.ny),self.h2/(self.ny))) and between(x[2], (-self.h3/(self.nz),self.h3/(self.nz)))

    def create_Gauss_point_querry_domain(self,mesh):
        """
        Create subdomains by marking regions
        """
        GaussDomain = MeshFunction("size_t", mesh, mesh.topology().dim())
        GaussDomain.set_all(0) #assigns material/props number 0 everywhere
        GaussDomainQuerry= self.Gauss_point_Querry(self.h1,self.h2,self.h3,self.nx,self.ny,self.nz)
        GaussDomainQuerry.mark(GaussDomain,1)
        return GaussDomain

    
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
        tmp=np.zeros(18)
        tmp[1-1]=self.Normal_loading_total
        tmp[5-1]=self.Normal_loading_total
        tmp[9-1]=self.Normal_loading_total

        self.feobj.sigma2.interpolate(Constant(tmp))
        tmp=np.zeros(6)
        self.feobj.usol.interpolate(Constant(tmp))
                    
        pass
        
    def set_bcs(self):
        """
        Set boundary conditions for the user problem / could be replaced by external mesher, e.g. Abaqus, Gmsh...
        """

        '''
        here a point load is applied in region 7
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
        self.u1_tot=0.1
        scale_u=1.#1./1000.

        bcs=[]
        if self.problem_step == 0:
            bcs=[
                [1,[0,[0,2],0.]],
                [2,[0,[0,2],0.]],
                [3,[0,[0,0],0.]],
                [4,[0,[0,0],0.]],
                [5,[0,[0,1],0.]],
                [6,[0,[0,1],0.]],
                
                [1,[0,[1,2],0.]],
                [2,[0,[1,2],0.]],
                [3,[0,[1,0],0.]],
                [4,[0,[1,0],0.]],
                [5,[0,[1,1],0.]],
                [6,[0,[1,1],0.]],
                ]        

        elif self.problem_step >= 1:
             bcs=[
                [1,[0,[0,1],self.u1_tot/scale_u]],
                [1,[0,[0,0],0.]],
                # [1,[3,[0,2],self.Normal_loading_total*scale_u]],
                [1,[0,[0,2],0]],


                [2,[0,[0,1],0.]],
                [2,[0,[0,0],0.]],
                [2,[0,[0,2],0.]],
                                
                [1,[0,[1,2],0.]],
                [2,[0,[1,2],0.]],
                [3,[0,[1,0],0.]],
                [4,[0,[1,0],0.]],
                [5,[0,[1,1],0.]],
                [6,[0,[1,1],0.]],
                ]        

        return bcs


    def history_output(self):
        """
        Used to get output of residual at selected node 
        """
        hist=[[2,[1,[0,0]]],
              [2,[0,[0,0]]],
              [2,[1,[0,1]]],
              [2,[0,[0,1]]],
              [2,[1,[0,2]]],
              [2,[0,[0,2]]],
              [2,[1,[1,0]]],
              [2,[0,[1,0]]],
              [2,[1,[1,1]]],
              [2,[0,[1,1]]],
              [2,[1,[1,2]]],
              [2,[0,[1,2]]],
 
              [4,[1,[0,0]]],
              [4,[0,[0,0]]],
              [4,[1,[0,1]]],
              [4,[0,[0,1]]],
              [4,[1,[0,2]]],
              [4,[0,[0,2]]],
              [4,[1,[1,0]]],
              [4,[0,[1,0]]],
              [4,[1,[1,1]]],
              [4,[0,[1,1]]],
              [4,[1,[1,2]]],
              [4,[0,[1,2]]],
 
              [6,[1,[0,0]]],
              [6,[0,[0,0]]],
              [6,[1,[0,1]]],
              [6,[0,[0,1]]],
              [6,[1,[0,2]]],
              [6,[0,[0,2]]],
              [6,[1,[1,0]]],
              [6,[0,[1,0]]],
              [6,[1,[1,1]]],
              [6,[0,[1,1]]],
              [6,[1,[1,2]]],
              [6,[0,[1,2]]],
              ]
        return hist

    def history_svars_output(self):
        """
        Used to get output of svars at selected Gauss point 
        """
        hist_svars=[[1,[1,[0]]], #sigma_11
                    [1,[1,[1]]], #sigma_12
                    [1,[1,[2]]], #sigma_13
                    [1,[1,[3]]], #sigma_21
                    [1,[1,[4]]], #sigma_22
                    [1,[1,[5]]], #sigma_23
                    [1,[1,[6]]], #sigma_31
                    [1,[1,[7]]], #sigma_32
                    [1,[1,[8]]], #sigma_33
                    [1,[1,[9]]], #mu_11
                    [1,[1,[10]]], #mu_12
                    [1,[1,[11]]], #mu_13
                    [1,[1,[12]]], #mu_21
                    [1,[1,[13]]], #mu_22
                    [1,[1,[14]]], #mu_23
                    [1,[1,[15]]], #mu_31
                    [1,[1,[16]]], #mu_32
                    [1,[1,[17]]], #mu_33
                    [1,[1,[18]]], #epsilon_11
                    [1,[1,[19]]], #epsilon_12
                    [1,[1,[20]]], #epsilon_13
                    [1,[1,[21]]], #epsilon_21
                    [1,[1,[22]]], #epsilon_22
                    [1,[1,[23]]], #epsilon_23
                    [1,[1,[24]]], #epsilon_31
                    [1,[1,[25]]], #epsilon_32
                    [1,[1,[26]]], #epsilon_33
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
        
        env_lib=ngeo_parameters.env_lib        #umat_lib='./libplast_Cauchy3D-DP.so'
        umat_lib_path= ngeo_parameters.umat_lib_path
        umat_lib = umat_lib_path+'/COSSERAT3D/libplast_Cosserat3D.so'
        # print("hello",os.path.abspath(umat_lib_path))
        umat_id=3      # if many materials exist in the same library
        mat=UserMaterial(env_lib,umat_lib,umat_id)
        mat.props=self.set_material_1_properties()
        #
        mats.append(mat)
        return mats
    
    def set_material_1_properties(self):
        """
        Sets material parameters
        """
        
        g1=8./5.;g2=2./5.;g3=8./5.;g4=2./5.
        
        h1=2./3. ;h2=-1./6.;h3=2./3.;h4=-1./6.;
        
        K=666.66; G=1.*10.**3.; Gc=0.5*10.**3. ; L=1*10**(2.);R=10.*10.**(-3.);  
        MG=G*(R**2.)/h3 ; MGc=MG; #th_g=0.; 
        tanfi=0.5; cc=0.#*1000. #*0000000.;
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
        saveto=reference_data_path+"Cosserat_3D_Drucker-Prager_test_step_0"+"_App_1"+".xdmf"
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
            filename = 'Cosserat_3D_Drucker-Prager_test_step_'+str(i+1)
            saveto= reference_data_path+"Cosserat_3D_Drucker-Prager_test_step_"+str(i+1)+"_App_1"+".xdmf"
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
    
    