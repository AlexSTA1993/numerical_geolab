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

warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
#from Parametric_Cosserat import Cosserat_1D_FEformulation

class THM1D_FEformulation(FEformulation):
    '''
    Defines a user FE formulation
    '''
    def __init__(self):
        # Number of stress/deformation components
        self.p_nstr=3+1+1+1
        # Number of Gauss points
        self.ns=1
        # Number of auxiliary quantities at gauss points
        self.p_aux=3
        self.count =0
        
    def generalized_epsilon(self,v):
        '''
        Set user's generalized deformation vector
        '''
        scale=1./1000.
        gde=[
            Dx(scale*v[0],0),              #gamma_11
            scale*v[2],                    #gamma_12
            Dx(scale*v[1],0)-scale*v[2],         #gamma_21
            Dx(scale*v[2],0),              #kappa_31
            Dx(v[3],0),  #q_1 - pf
            Dx(v[4],0),  #q_1 - temp
            ]

        return as_vector(gde)
    
    def auxiliary_fields(self,v):
        '''
        Set user's generalized deformation vector
        '''
        auxgde=[
        v[3],
        v[4],
        v[0]
            ]
        return as_vector(auxgde)
    
    def setVarFormAdditionalTerms_Res(self,u,Du,v,svars,metadata,dt):
        Res=0.
        lstar=svars.sub(103-1)
        bstar=svars.sub(104-1)
        rhoC=svars.sub(105-1)
        #HM terms
        eps=self.generalized_epsilon(Du)
        eps_v=eps[0]
        virtual_pf=v[3]
             
        Res+=-(1./dt)*(1./bstar)*dot(eps_v,virtual_pf)*dx(metadata=metadata) 
             
        #TM terms
        virtual_Temp=v[4]
        for i in range(1,18):
            Res+= + (1./dt)*(1./rhoC)*svars.sub(i-1)*svars.sub(76+i-1)*virtual_Temp*dx(metadata=metadata)
        #HT terms
        DTemp=Du[4]
        Res+= +(1./dt)*(lstar/bstar)*dot(DTemp,virtual_pf)*dx(metadata=metadata)
           
        return Res
    
    def setVarFormAdditionalTerms_Jac(self,u,Du,v,svars,metadata,dt,ddsdde):
        if self.count==0:
            self.ddsdde_el = ddsdde
#             print(self.ddsdde_el.vector().get_local())
#             print(self.ddsdde_el)
            self.count = 1 
        Jac=0.
        lstar=svars.sub(101+2-1)
        bstar=svars.sub(102+2-1)
        rhoC=svars.sub(103+2-1)
        alfa=svars.sub(104+2-1)
        #HM terms
        eps=self.generalized_epsilon(u) #needs u (trial function, because it takes derivatives in terms of u and not Du for calculating the Jacobian.
        eps_vol=eps[0]
        virtual_pf=v[3]
        Jac+=+(1./dt)*(1./bstar)*dot(eps_vol,virtual_pf)*dx(metadata=metadata)
         
        #MH terms
        pf=u[3] #same as before
        virtual_eps=self.generalized_epsilon(v)
        virtual_eps_vol=virtual_eps[0]
        Jac+=-(1./dt)*dt*dot(pf,virtual_eps_vol)*dx(metadata=metadata)
                 
        #HT terms
        temperature = u[4]
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
        virtual_temp=v[4]
#         eps_eff=eps+eps_temp 
#         eps_eff=as_vector([eps[0],eps[1],eps[2],eps[3],0,0]) #Changed in December
#         epl_eff =as_vector([eps[0],eps[1],eps[2],eps[3],0,0])
        eps_eff=as_vector([0,eps[1],eps[2],eps[3],0,0]) #Changed in December
        epl_eff =as_vector([0,eps[1],eps[2],eps[3],0,0])
        deps_plastic=[]
        eps_elastic = [] #added on December
        # small fix December 2/12 for the plastic strains
#         for i in range(0,self.p_nstr):
#             deps_plastic.append(svars.sub(77-1+i))
        deps_plastic.append(svars.sub(77-1))
        deps_plastic.append(svars.sub(78-1))
        deps_plastic.append(svars.sub(80-1))
        deps_plastic.append(svars.sub(92-1))
        deps_plastic.append(0)
        deps_plastic.append(0)
        
        eps_elastic.append(svars.sub(25-1)-svars.sub(50-1)-svars.sub(102-1)*svars.sub(104+2-1))
        eps_elastic.append(svars.sub(26-1)-svars.sub(51-1))
        eps_elastic.append(svars.sub(28-1)-svars.sub(53-1))
        eps_elastic.append(svars.sub(40-1)-svars.sub(65-1))
        eps_elastic.append(0)
        eps_elastic.append(0)
        
        deps_plastic=as_vector(deps_plastic)
        eps_elastic = as_vector(eps_elastic)
#         Jac+=-(1./dt)*dt*(1./rhoC)*inner(dot(ddsdde,eps_eff),deps_plastic)*virtual_temp*dx(metadata=metadata)
#         Jac+=(1./dt)*(1./rhoC)*inner(dot(ddsdde_el,eps_eff),deps_plastic)*virtual_temp*dx(metadata=metadata)
#         Jac+=(1./dt)*(1./rhoC)*inner(dot(self.ddsdde_el,eps_eff),deps_plastic)*virtual_temp*dx(metadata=metadata) #Changed on December 2
#         Jac+=(1./dt)*(1./rhoC)*inner(dot(self.ddsdde_el,eps_elastic),epl_eff)*virtual_temp*dx(metadata=metadata) #Changed on December 2

        #TM terms due to fluid pressure and plastic deforamtion, i.e. thermal pressurization 
#         deps_plastic_vol=deps_plastic[0]
# #         Jac+=+(1./dt)*dt*(1./rhoC)*pf*deps_plastic_vol*virtual_temp*dx(metadata=metadata)
#         Jac+=+(1./dt)*(1./rhoC)*pf*deps_plastic_vol*virtual_temp*dx(metadata=metadata)

             
        #TM Alexandros
#         Jac+=(1./dt)*dt*(1./rhoC)*inner(dot(ddsdde,eps_temp),deps_plastic)*virtual_temp*dx(metadata=metadata)
#         Jac+=-(1./dt)*(1./rhoC)*inner(dot(ddsdde,eps_temp),deps_plastic)*virtual_temp*dx(metadata=metadata) #change sign
        return Jac
#         def setVarFormAdditionalTerms_Res(self,u,Du,v,svars,metadata,dt):
#             Res=0.
#             lstar=svars.sub(103-1)
#             bstar=svars.sub(104-1)
#             rhoC=svars.sub(105-1)
#         #HM terms
#             eps=self.generalized_epsilon(Du)
#             eps_v=eps[0]
#             virtual_pf=v[3]
#              
#             Res+=-(1./dt)*(1./bstar)*dot(eps_v,virtual_pf)*dx(metadata=metadata) 
#              
#         #TM terms
#             virtual_Temp=v[4]
#             for i in range(1,18):
#                 Res+= + (1./dt)*(1./rhoC)*svars.sub(i-1)*svars.sub(76+i-1)*virtual_Temp*dx(metadata=metadata)
#         #HT terms
#             DTemp=Du[4]
#             Res+= +(1./dt)*(lstar/bstar)*dot(DTemp,virtual_pf)*dx(metadata=metadata)
#            
#             return Res
#     
#         def setVarFormAdditionalTerms_Jac(self,u,Du,v,svars,metadata,dt,ddsdde):
#             Jac=0.
#             lstar=svars.sub(101+2-1)
#             bstar=svars.sub(102+2-1)
#             rhoC=svars.sub(103+2-1)
#             alfa=svars.sub(104+2-1)
#         #HM terms
#             eps=self.generalized_epsilon(u) #needs u (trial function, because it takes derivatives in terms of u and not Du for calculating the Jacobian.
#             eps_vol=eps[0]
#             virtual_pf=v[3]
#             Jac+=+(1./dt)*(1./bstar)*dot(eps_vol,virtual_pf)*dx(metadata=metadata)
#          
#         #MH terms
#             pf=u[3] #same as before
#             virtual_eps=self.generalized_epsilon(v)
#             virtual_eps_vol=virtual_eps[0]
#             Jac+=-(1./dt)*dt*dot(pf,virtual_eps_vol)*dx(metadata=metadata)
#                  
#         #HT terms
#             temperature = u[4]
#             Jac+=-(1./dt)*(lstar/bstar)*dot(temperature,virtual_pf)*dx(metadata=metadata)
#                 
#         #TH terms Alexandros
#             avector=np.zeros(self.p_nstr)
#             avector[0]=1.;
#             eps_temp=alfa*temperature*as_vector(avector)
# #         eps_temp_vol=eps_temp[0]+eps_temp[1]+eps_temp[2]
# #         Jac+=(1./bstar)*dot(eps_temp_vol,virtual_pf)*dx(metadata=metadata) 
# #         
#         #MT terms
#         #eps_temp=alfa*temperature*as_vector([1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
#             Jac+=-(1./dt)*dt*inner(dot(ddsdde,eps_temp),virtual_eps)*dx(metadata=metadata) 
#                 
#         #TM terms due to thermal expansion and plastic deformation
#             virtual_temp=v[4]
#             eps_eff=eps+eps_temp
#             deps_plastic=[]
#             for i in range(0,self.p_nstr):
#                 deps_plastic.append(svars.sub(77-1+i))
#           
#             deps_plastic=as_vector(deps_plastic)
#             Jac+=-(1./dt)*dt*(1./rhoC)*inner(dot(ddsdde,eps_eff),deps_plastic)*virtual_temp*dx(metadata=metadata)
# #         Jac+=-(1./dt)*(1./rhoC)*inner(dot(ddsdde,eps_eff),deps_plastic)*virtual_temp*dx(metadata=metadata)
#         #TM terms due to fluid pressure and plastic deforamtion, i.e. thermal pressurization 
#             deps_plastic_vol=deps_plastic[0]
#             Jac+=+(1./dt)*dt*(1./rhoC)*pf*deps_plastic_vol*virtual_temp*dx(metadata=metadata)
# #         Jac+=+(1./dt)*(1./rhoC)*pf*deps_plastic_vol*virtual_temp*dx(metadata=metadata)
# 
#              
#         #TM Alexandros
#             Jac+=(1./dt)*dt*(1./rhoC)*inner(dot(ddsdde,eps_temp),deps_plastic)*virtual_temp*dx(metadata=metadata)
# #         Jac+=(1./dt)*(1./rhoC)*inner(dot(ddsdde,eps_temp),deps_plastic)*virtual_temp*dx(metadata=metadata)
#             return Jac    

    
    def create_element(self,cell):
        """
        Set desired element
        """
        self.degree=1
#         element=VectorElement("Lagrange",cell,degree=self.degree,dim=8)
        element1=VectorElement("Lagrange",cell,degree=self.degree,dim=2)
        element2=FiniteElement("Lagrange",cell,degree=self.degree)
        element3=VectorElement("Lagrange",cell,degree=self.degree,dim=2)

        element=MixedElement([element1,element2,element3])
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
        return as_vector([0.,0.,0.,1.,1.])

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
        self.genprops.p_nsvars=110
        
    def create_mesh(self):
        """
        Set mesh and subdomains
        """
#         pt=Point(0.,0.,100.)
#         pb=Point(0.,0.,0.)
#         geometry=Cylinder(pt,pb,25.,25.)
#         mesh=generate_mesh(geometry,32)
        self.h=10.
        self.n=100
       
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
                [1, [2, [0,0], 0.]],
                [1, [2, [0,1], 0.]],
                [1, [2, [1], 0.]],
                [1, [1, [2,0],0]],
                [1, [1, [2,1],1]],
#               #right
                [2, [1, [0,0], 0.]],
                [2, [1, [0,1],0.]],
                [2, [1, [1], 0.]],
                [2, [1, [2,0],0]],
                [2, [1, [2,1],0]],
                               
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
        umat_lib= umat_lib_path+'/COSSERAT3D-THM/libplast_Cosserat3D-THM.so'
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
        
        g1=8./5.;g2=2./5.;g3=8./5.;g4=2./5.
        
        h1=2./3. ;h2=-1./6.;h3=2./3.;h4=-1./6.;
        
        K=20.*10.**3.; G=10.*10.**3.; Gc=10.*10.**3. ; L=1*10**(3.);R=10.*10.**(-3.);  
        MG=G*(R**2.)/h3 ; MGc=MG; #th_g=0.; 
        permeability1 = 1.;fluid_viscocity = 1.;bstar=10**16;
        permeability = permeability1/bstar
        conductivity = 1.; rhoC = 1; alpha =0.*10.**-5.; lstar = 10.**16;
        tanfi=0.; cc=10000000.#*0000000.;
        tanpsi=0.; Hsfi=0.; Hscc=0.;
        
        
        eta1=0.
        
        prop_num=29
        props=np.zeros(prop_num)
        props[0]=K
        props[1]=G
        props[2]=Gc
        props[3]=L
        props[4]=MG
        props[5]=MGc
        props[6]=R
        props[7]=permeability
        props[8]=fluid_viscocity
        props[9]=bstar
        props[10]=conductivity
        props[11]=rhoC
        props[12]=alpha
        props[13]=lstar
        props[14]=0.
        props[15]=tanfi
        props[16]=cc
        props[17]=tanpsi
        props[18]=Hsfi
        props[19]=Hscc
        props[20]=h1
        props[21]=h2
        props[22]=h3
        props[23]=h4
        props[24]=g1
        props[25]=g2
        props[26]=g3
        props[27]=g4
        props[28]=eta1
        
        return props
    
    def history_output(self):
        history_output=[
            [2,[0,[2,0]]],
            [2,[0,[2,1]]]
            ]
        return history_output
    
    def plot_me(self):
        self.time=[0]
        self.pressure=[0.]
        self.temperature=[0.]
#         print(self.feobj.problem_history)
        for g in self.feobj.problem_history:
            if int(g[0]) % int(10) == int(0):
                self.time.append(g[0])
                self.pressure.append(g[2][0])
                self.temperature.append(g[2][1])
#         print(a)    
#         print(b)
        self.temperature = np.array(self.temperature) 
        fig, ax = plt.subplots()
        ax.plot(self.time,self.pressure,markersize=6,color='blue',marker='o',label ='pressure at free end')
        ax.plot(self.time,self.temperature,markersize=6,color='orange',marker='+',label ='temperature at free end')
        ax.set(xlabel='time (s)', ylabel='Pressure MPa',title='Pressure versus time at right end')
        ax.grid()
        ax.legend(loc='lower right')
        plt.show()
    
    def give_me_solver_params(self):
        reference_data_path = ngeo_parameters.reference_data_path   
        import os
#         nel=self.n
        self.slv.dtmax=10.
        self.slv.tmax=500.
#         ninc=int(self.slv.tmax/self.slv.dtmax)
        filename='test1D_HydroThermal'
#         a=os.path.join(reference_data_path,filename+'.xdmf')
#         self.saveto=a
        self.saveto=reference_data_path+filename+'.xdmf'
        self.slv.nincmax=10000
        self.slv.convergence_tol=10**-4
        self.slv.removezerolines=False
    
    def pressure_vs_temperature(self):
        svars = np.reshape(self.feobj.svars2.vector().get_local(),(-1,self.feobj.p_nsvars))
        diff = svars[:,[100]]-svars[:,[101]]
        print(svars[:,[100]])
        print(svars[:,[101]])
        return diff  
      
# my_FEformulation=THM1D_FEformulation()
# my_FEproblem=THM1D_FEproblem(my_FEformulation)
# my_FEproblem.give_me_solver_params()
# converged=my_FEproblem.solve(my_FEproblem.saveto,summary=True)
# my_FEproblem.plot_me()
# my_FEproblem.pressure_vs_temperature()

#OLD TERMS RES/JAC
#     def setVarFormAdditionalTerms_Res(self,u,Du,v,svars,metadata,dt):
#         Res=0.
#         lstar=svars.sub(103-1)
#         bstar=svars.sub(104-1)
#         rhoC=svars.sub(105-1)
#         #HM terms
#         eps=self.generalized_epsilon(Du)
#         eps_v=eps[0]
#         virtual_pf=v[3]
#            
#         Res+=-(1./bstar)*dot(eps_v,virtual_pf)*dx(metadata=metadata) 
#            
#         #TM terms
#         virtual_Temp=v[4]
#         for i in range(1,18):
#             Res+= + (1./rhoC)*svars.sub(i-1)*svars.sub(76+i-1)*virtual_Temp*dx(metadata=metadata)
#         #HT terms
#         DTemp=Du[4]
#         Res+= +(lstar/bstar)*dot(DTemp,virtual_pf)*dx(metadata=metadata)
#           
#         return Res
#     
#     def setVarFormAdditionalTerms_Jac(self,u,Du,v,svars,metadata,dt,ddsdde):
#         Jac=0.
#         lstar=svars.sub(101+2-1)
#         bstar=svars.sub(102+2-1)
#         rhoC=svars.sub(103+2-1)
#         alfa=svars.sub(104+2-1)
#         #HM terms
#         eps=self.generalized_epsilon(u) #needs u (trial function, because it takes derivatives in terms of u and not Du for calculating the Jacobian.
#         eps_vol=eps[0]
#         virtual_pf=v[3]
#         Jac+=+(1./bstar)*dot(eps_vol,virtual_pf)*dx(metadata=metadata)
#     
#         #MH terms
#         pf=u[3] #same as before
#         virtual_eps=self.generalized_epsilon(v)
#         virtual_eps_vol=virtual_eps[0]
#         Jac+=-dt*dot(pf,virtual_eps_vol)*dx(metadata=metadata)
#             
#         #HT terms
#         temperature = u[4]
#         Jac+=-(lstar/bstar)*dot(temperature,virtual_pf)*dx(metadata=metadata)
#            
#         #TH terms Alexandros
#         avector=np.zeros(self.p_nstr)
#         avector[0]=1.;
#         eps_temp=alfa*temperature*as_vector(avector)
# #         eps_temp_vol=eps_temp[0]+eps_temp[1]+eps_temp[2]
# #         Jac+=(1./bstar)*dot(eps_temp_vol,virtual_pf)*dx(metadata=metadata) 
# #         
#         #MT terms
#         #eps_temp=alfa*temperature*as_vector([1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
#         Jac+=-dt*inner(dot(ddsdde,eps_temp),virtual_eps)*dx(metadata=metadata) 
#            
#         #TM terms due to thermal expansion and plastic deformation
#         virtual_temp=v[4]
#         eps_eff=eps+eps_temp
#         deps_plastic=[]
#         for i in range(0,self.p_nstr):
#             deps_plastic.append(svars.sub(76-1+i))
#         deps_plastic=as_vector(deps_plastic)
#         Jac+=-dt*(1./rhoC)*inner(dot(ddsdde,eps_eff),deps_plastic)*virtual_temp*dx(metadata=metadata)
#         #TM terms due to fluid pressure and plastic deforamtion, i.e. thermal pressurization 
#         deps_plastic_vol=deps_plastic[0]
#         Jac+=+dt*(1./rhoC)*pf*deps_plastic_vol*virtual_temp*dx(metadata=metadata)
#         print("hello!!!")   
#         #TM Alexandros
#         Jac+=dt*(1./rhoC)*inner(dot(ddsdde,eps_temp),deps_plastic)*virtual_temp*dx(metadata=metadata)
