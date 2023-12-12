'''
Created on Dec 6, 2019

@author: alexandrosstathas
'''

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
from ngeoFE_unittests import ngeo_parameters
# ngeo_parameters.reference_data_path='/home/alexandrosstathas/eclipse-workspace/numerical_geolab/Numerical_Geolab/ngeoFE_unittests/Multiphysics/reference_data/'

# from 0test_cauchy1Dmat import svars
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
#from Parametric_Cosserat import Cosserat_1D_FEformulation

class THM1D_FEformulation(FEformulation):
    '''
    Defines a user FE formulation
    '''
    def __init__(self):
        # Number of stress/deformation components
        self.p_nstr=2+1+1
        # Number of Gauss points
        self.ns=1
        # Number of auxiliary quantities at gauss points
        self.p_aux=3
        self.count =0
        
    def generalized_epsilon(self,v):
        '''
        Set user's generalized deformation vector
        '''
        scale=1./1.
        gde=[
            Dx(scale*v[0],0),              #gamma_11
            Dx(scale*v[1],0),         #gamma_21
            Dx(v[2],0),  #q_1 - pf
            Dx(v[3],0),  #q_1 - temp
            ]

        return as_vector(gde)
    
    def auxiliary_fields(self,v):
        '''
        Set user's generalized deformation vector
        '''
        auxgde=[
        v[2],
        v[3],
        v[0]
            ]
        return as_vector(auxgde)
    
    def setVarFormAdditionalTerms_Res(self,u,Du,v,svars,metadata,dt):
        Res=0.
        lstar=svars.sub(55-1)
        bstar=svars.sub(56-1)
        rhoC=svars.sub(57-1)
        #HM terms
        eps=self.generalized_epsilon(Du)
        eps_v=eps[0]
        virtual_pf=v[2]
              
        Res+=-(1./dt)*(1./bstar)*dot(eps_v,virtual_pf)*dx(metadata=metadata) 
              
        #TM terms
        virtual_Temp=v[3]
        for i in range(1,7):
            Res+= + (1./dt)*(1./rhoC)*svars.sub(i-1)*svars.sub(40+i-1)*virtual_Temp*dx(metadata=metadata)
        #HT terms
        DTemp=Du[3]
        Res+= +(1./dt)*(lstar/bstar)*dot(DTemp,virtual_pf)*dx(metadata=metadata)
           
        return Res
    
    def setVarFormAdditionalTerms_Jac(self,u,Du,v,svars,metadata,dt,ddsdde):
        if self.count==0:
            self.ddsdde_el = ddsdde
#             print(self.ddsdde_el.vector().get_local())
#             print(self.ddsdde_el)
            self.count = 1 
        Jac=0.
        lstar=svars.sub(55-1)
        bstar=svars.sub(56-1)
        rhoC=svars.sub(57-1)
        alfa=svars.sub(58-1)
        #HM terms
        eps=self.generalized_epsilon(u) #needs u (trial function, because it takes derivatives in terms of u and not Du for calculating the Jacobian.
        eps_vol=eps[0]
        virtual_pf=v[2]
        Jac+=+(1./dt)*(1./bstar)*dot(eps_vol,virtual_pf)*dx(metadata=metadata)
          
        #MH terms
        pf=u[2] #same as before
        virtual_eps=self.generalized_epsilon(v)
        virtual_eps_vol=virtual_eps[0]
        Jac+=-(1./dt)*dt*dot(pf,virtual_eps_vol)*dx(metadata=metadata)
                  
        #HT terms
        temperature = u[3]
        Jac+=-(1./dt)*(lstar/bstar)*dot(temperature,virtual_pf)*dx(metadata=metadata)
                 
        #TH terms Alexandros
        avector=np.zeros(self.p_nstr)
        avector[0]=1.;
        eps_temp=alfa*temperature*as_vector(avector)
        eps_temp_vol=eps_temp[0]#+eps_temp[1]+eps_temp[2]
#Corection December 2/12 no need for this term
#         Jac+=(1./dt)*(1./bstar)*dot(eps_temp_vol,virtual_pf)*dx(metadata=metadata) #uncomment
#         
        #MT terms
        #eps_temp=alfa*temperature*as_vector([1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        Jac+=-(1./dt)*dt*inner(dot(ddsdde,eps_temp),virtual_eps)*dx(metadata=metadata) #changed sign
                 
        #TM terms due to thermal expansion and plastic deformation
        virtual_temp=v[3]
#         eps_eff=eps+eps_temp 
#         eps_eff=as_vector([eps[0],eps[1],eps[2],eps[3],0,0]) #Changed in December
#         epl_eff =as_vector([eps[0],eps[1],eps[2],eps[3],0,0])
         
        #change in Jacobian terms
         
        eps_eff=eps+eps_temp
        deps_plastic=[]
        for i in range(0,self.p_nstr):
            deps_plastic.append(svars.sub(41-1+i))
           
        deps_plastic=as_vector(deps_plastic)
        Jac+=-(1./dt)*dt*(1./rhoC)*inner(dot(ddsdde,eps_eff),deps_plastic)*virtual_temp*dx(metadata=metadata)
#         Jac+=-(1./dt)*(1./rhoC)*inner(dot(ddsdde,eps_eff),deps_plastic)*virtual_temp*dx(metadata=metadata)
        #TM terms due to fluid pressure and plastic deforamtion, i.e. thermal pressurization 
        deps_plastic_vol=deps_plastic[0]
        Jac+=+(1./dt)*dt*(1./rhoC)*pf*deps_plastic_vol*virtual_temp*dx(metadata=metadata)
#         Jac+=+(1./dt)*(1./rhoC)*pf*deps_plastic_vol*virtual_temp*dx(metadata=metadata)
 
              
        #TM Alexandros
        Jac+=(1./dt)*dt*(1./rhoC)*inner(dot(ddsdde,eps_temp),deps_plastic)*virtual_temp*dx(metadata=metadata)
        return Jac

    
    def create_element(self,cell):
        """
        Set desired element
        """
        self.degree=1
#         element=VectorElement("Lagrange",cell,degree=self.degree,dim=8)
        element=VectorElement("Lagrange",cell,degree=self.degree,dim=4)
#         element3=VectorElement("Lagrange",cell,degree=self.degree,dim=2)

#         element=MixedElement([element1,element3])
#         self.degree=2
#         # Defines a Lagrangian FE of degree n for the displacements
#         element_disp=VectorElement("Lagrange",cell,degree=self.degree,dim=3)
#         # Defines a Lagrangian FE of degree n-1 for the rotations 
#         if self.degree-1 != 0:
#             element_rot=VectorElement("Lagrange",cell,degree=2,dim=3)
#         else:
#             element_rot=VectorElement("Lagrange",cell,degree=2,dim=3)
#         # Defines a Langrangian FE of degree n for pressure and temperature
#         element_pt=VectorElement("Lagrange",cell,degree=self.degree,dim=2)
#         element=MixedElement([element_disp,element_rot,element_pt])
        return element

    def dotv_coeffs(self):
        """   
        Set left hand side derivative coefficients
        """
        return as_vector([0.,0.,1.,1.])

class THM1D_FEproblem(UserFEproblem):
    def __init__(self,FEformulation):
        self.description="Example of 1D plane strain problem, Cauchy continuum"
        self.h = 1.
        super().__init__(FEformulation)
    
    def set_general_properties(self):
        """
        Set here all the parameters of the problem, except material properties 
        """
        self.genprops=General_FEproblem_properties()
        # Number of state variables
        self.genprops.p_nsvars=63
        
    def create_mesh(self):
        """
        Set mesh and subdomains
        """
#         pt=Point(0.,0.,100.)
#         pb=Point(0.,0.,0.)
#         geometry=Cylinder(pt,pb,25.,25.)
#         mesh=generate_mesh(geometry,32)
        self.h=60.
        self.n=1
       
        mesh=IntervalMesh(self.n,-self.h/2.,self.h/2.)
#         mesh=BoxMesh(Point(-0.5*self.h1,-0.5*self.h2,-0.5*self.h3),Point(0.5*self.h1,0.5*self.h2,0.5*self.h3),self.nx,self.ny,self.nz)
        #import matplotlib.pyplot as plt
        #plot(mesh, title="cubic mesh", wireframe=True)
        #plt.show()
        
      
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
    
    def mark_boundaries(self, boundaries):
        """
        Mark left and right boundary points
        """
        left0 = self.Boundary(0,-self.h/2.)
        left0.mark(boundaries, 1)
        right0 = self.Boundary(0,self.h/2.)
        right0.mark(boundaries, 2)
        #         
        return
  
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
#         u_displ=-0.
#         p_f=0.
#         Temp=1.
        bcs = [    

#               #left
                [1, [0, [0], 0.]],
                [1, [0, [1], 0.]],
                [1, [1, [2],0.]],
                [1, [1, [3],0.]],
#               #right
                [2, [0, [0], 0.0]],
                [2, [0, [1], 2.0]],
                [2, [1, [2],0.]],
                [2, [1, [3],0.]]
                
                #left
#                 [1, [2, [0,0], 0.]],
#                 [1, [0, [0,1], 0.]],
#                 [1, [1, [1], 0.]],
#                 [1, [1, [2,0],0]],
#                 [1, [1, [2,1],0]],
# #               #right
#                 [2, [0, [0,0], -1000.]],
#                 [2, [0, [0,1],0.]],
#                 [2, [1, [1], 0.]],
#                 [2, [1, [2,0],0]],
#                 [2, [1, [2,1],0]],
            ]        
        return bcs
     
    def set_materials(self):
        """
        Create material objects and set material parameters
        """
        mats=[]
        # load material #1
        env_lib=ngeo_parameters.env_lib        #umat_lib='./libplast_Cauchy3D-DP.so'
        umat_lib_path= ngeo_parameters.umat_lib_path
#         env_lib=['/usr/lib/x86_64-linux-gnu/lapack/liblapack.so']
#         umat_lib='/mnt/f/DEVELOPMENT/Numerical_Geolab_Materials-F/UMATERIALS/COSSERAT3D-THM/libplast_Cosserat3D-THM.so'
        umat_lib= umat_lib_path+'CAUCHY3D-DP-PR-TEMP/libplast_Cauchy3D-DP-PR-TEMP.so'
        umat_id=1 # if many materials exist in the same library
        mat=UserMaterial(env_lib,umat_lib,umat_id)
        mat.props=self.set_material_1_properties()
        mats.append(mat)
        return mats
    
    def set_material_1_properties(self):
        """
        Sets material parameters
        """
#         EE=3.*9./7.;nu=2./7. ;cc=10000000.;tanfi=0.;tanpsi=0.;Hsfi=0.;Hscc=0.
#         GG=EE/(2.*(1.+nu))
#         KK=EE*GG/(3.*(3.*GG-EE))
#         permeability=1.;fluid_viscosity=1.;lstar=1.;bstar=10.**8;alpha=2.5*10.**-2.
#         conductivity=1.;rhoC=1. #density * specific heat capacity
#         props=np.array([KK,GG,permeability,fluid_viscosity,bstar,conductivity,rhoC,alpha,lstar,0.,tanfi,cc,tanpsi,Hsfi,Hscc,0.,0.,0.,0.,0.])
#         props=props.astype("double")
        
        K=4000.; G=4000.;
        permeability1 = 1.;fluid_viscocity = 1.;bstar=1*10**8.;
        permeability = permeability1/bstar
        conductivity = 1.*10**-8; rhoC = 1; alpha =0.*10.**-13.; lstar = 1*10**8.;
        tanfi=0.; cc=0.1#*0000000.;
        tanpsi=0.; Hsfi=0.; Hscc=0.;
        
        
        eta1=0.
        
        prop_num=19
        props=np.zeros(prop_num)
        props[0]=K
        props[1]=G
        props[2]=permeability
        props[3]=fluid_viscocity
        props[4]=bstar
        props[5]=conductivity
        props[6]=rhoC
        props[7]=alpha
        props[8]=lstar
        props[9]=0.
        props[10]=tanfi
        props[11]=cc
        props[12]=tanpsi
        props[13]=Hsfi
        props[14]=Hscc
        props[18]=eta1
        
        return props
    
#     def history_output(self):
#         history_output=[
#             [2,[0,[1]]],
#             [2,[0,[2]]],
#             [2,[0,[3]]]
#             ]
#         return history_output
    
    def plot_me(self):
#         V = VectorFunctionSpace(self.mesh,'Lagrange',1,dim = self.feobj.p_nsvars)
#         svars_nodes = project(self.feobj.svars2,V)
        self.time=[0]
        self.epsilon =[0.]
        self.pressure=[0.]
        self.temperature=[0.]
#         print(self.feobj.problem_history)
        for g in self.feobj.problem_history:
            if int(g[0]) % int(10) == int(0):
                self.time.append(g[0])
                self.pressure.append(g[2][1])
                self.temperature.append(g[2][2])
                self.epsilon.append(g[2][0]/self.h)
#         print(a)    
#         print(b)
        self.temperature = np.array(self.temperature) 
        fig, ax = plt.subplots()
        ax.plot(self.time,self.pressure,markersize=6,color='blue',marker='o',label ='pressure at free end')
        ax.plot(self.time,self.temperature,markersize=6,color='orange',marker='+',label ='temperature at free end')
        ax.set(xlabel='time (s)', ylabel='P MPa, T $^{0}$C',title='Pressure versus time at right end')
        ax.grid()
        ax.legend(loc='lower right')
        plt.show()
        fig, ax = plt.subplots()
        ax.plot(self.time,self.epsilon,markersize=4,color='red',marker='d',label ='volumetric strain at free end')
        ax.grid()
        ax.set(xlabel='time (s)', ylabel='$\epsilon^{tot}_{v}$',title='volumetric strain versus time at right end')
        ax.grid()
        ax.legend(loc='lower right')
        plt.show()
    def give_me_solver_params(self):
        reference_data_path = ngeo_parameters.reference_data_path   
        import os
#         nel=self.n
        self.slv.dtmax=1.
        self.slv.tmax=20.
#         ninc=int(self.slv.tmax/self.slv.dtmax)
        filename='test1D_ThermalPlastic_Cauchy_new'
#         a=os.path.join(reference_data_path,filename+'.xdmf')
#         self.saveto=a
        self.saveto=reference_data_path+filename+'.xdmf'
        self.slv.nincmax=10000
        self.slv.convergence_tol=10**-4
        self.slv.removezerolines=False
    
    def yield_criterion_particular_case(self):
        svars = np.reshape(self.feobj.svars2.vector().get_local(),(-1,self.feobj.p_nsvars))
        
        p = (svars[:,[0]]+svars[:,[1]]+svars[:,[2]])/3
        s11 = svars[:,[0]]-p
        s22 = svars[:,[1]]-p
        s33 = svars[:,[2]]-p
        s12 = svars[:,[5]]
        s13 = svars[:,[4]]
        s23 = svars[:,[3]]
        
        cc=100.
        cc = np.full((svars.shape[0],1), cc)
        
        Q1= 1./2.*(s11**2.+s22**2.+s33**2.+2.*s12**2.+2.*s13**2.+2.*s23**2.)

        tau = np.sqrt(Q1)
        diff = tau-cc
        
#         svars = np.reshape(self.feobj.svars2.vector().get_local(),(-1,self.feobj.p_nsvars))
        diff1 = svars[:,[0]]-svars[:,[1]]
        diff2 = svars[:,[0]]-svars[:,[2]]
        diff4 = svars[:,[5]] - cc
        for i in range(len(svars[:,[52]])):
            print(diff,svars[:,[0]],svars[:,[1]],svars[:,[2]],svars[:,[5]])
        return diff, diff1, diff2, diff4
    
    def dtemp_vs_plastic_work(self):
        svars = np.reshape(self.feobj.svars2.vector().get_local(),(-1,self.feobj.p_nsvars))
        plastic_work_rate = np.zeros(svars[:,[40]].shape)
        for j in range(0,6):
            plastic_work_rate +=svars[:,[40+j]]*svars[:,[0+j]] 
        for i in range(len(svars[:,[62]])):
            print(svars[i,[62]],plastic_work_rate[i])
            print(svars[i,[41]],svars[i,[42]])
        diff = svars[:,[62]]-plastic_work_rate[:]
        return diff
    
    def pressure_vs_temperature_vs_volumetric_strain(self):
        svars = np.reshape(self.feobj.svars2.vector().get_local(),(-1,self.feobj.p_nsvars))
        diff = svars[:,[52]]-svars[:,[53]]+svars[:,[12]]
        for i in range(len(svars[:,[52]])):
            print(svars[i,[52]],svars[i,[53]],svars[i,[12]])
        return diff  

if __name__ == "__main__":      
    my_FEformulation=THM1D_FEformulation()
    my_FEproblem=THM1D_FEproblem(my_FEformulation)
    my_FEproblem.give_me_solver_params()
    converged=my_FEproblem.solve(my_FEproblem.saveto,summary=True)
    my_FEproblem.plot_me()
    my_FEproblem.yield_criterion_particular_case()
    my_FEproblem.dtemp_vs_plastic_work()
    my_FEproblem.pressure_vs_temperature_vs_volumetric_strain()

