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

warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

class CosseratTHM1DFEformulation(FEformulation):
    '''
    Defines a user FE formulation
    '''
    def __init__(self):
        # Number of stress/deformation components
        self.p_nstr=4+1+1
        # Number of Gauss points
        self.ns=1
        # Number of auxiliary quantities at gauss points
        self.p_aux=3
    
    def generalized_epsilon(self,v):
        """
        Set user's generalized deformation vector
        """
        scale_u=1./1000.
        gde=[
            Dx(v[0],0)*scale_u,#gamma_11
            v[2]*scale_u,                    #gamma_12
            Dx(v[1],0)*scale_u-v[2]*scale_u,         #gamma_21
            Dx(v[2],0)*scale_u,              #kappa_31
            Dx(v[3],0),              #qhy_11
            Dx(v[4],0),              #qth_11
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
                
        #TH terms 
        avector=np.zeros(self.p_nstr)
        avector[0]=1.;
        eps_temp=alfa*temperature*as_vector(avector)
        eps_temp_vol=eps_temp[0]#+eps_temp[1]+eps_temp[2]
        #MT terms
        Jac+=-(1./dt)*dt*inner(dot(ddsdde,eps_temp),virtual_eps)*dx(metadata=metadata) #changed sign
                
        #TM terms due to thermal expansion and plastic deformation
        virtual_temp=v[4]
        
        #change in Jacobian terms
        
        eps_eff=eps+eps_temp
        deps_plastic=[]
        for i in range(0,self.p_nstr):
            deps_plastic.append(svars.sub(77-1+i))
          
        deps_plastic=as_vector(deps_plastic)
        Jac+=-(1./dt)*dt*(1./rhoC)*inner(dot(ddsdde,eps_eff),deps_plastic)*virtual_temp*dx(metadata=metadata)
        #TM terms due to fluid pressure and plastic deforamtion, i.e. thermal pressurization 
        deps_plastic_vol=deps_plastic[0]
        Jac+=+(1./dt)*dt*(1./rhoC)*pf*deps_plastic_vol*virtual_temp*dx(metadata=metadata)
             
        #TM 2
        Jac+=(1./dt)*dt*(1./rhoC)*inner(dot(ddsdde,eps_temp),deps_plastic)*virtual_temp*dx(metadata=metadata)
        return Jac
   
    def create_element(self,cell):
        """
        Set desired element
        """
        self.degree=1
        element1=VectorElement("Lagrange",cell,degree=self.degree,dim=2)
        element2=FiniteElement("Lagrange",cell,degree=self.degree)
        element3=VectorElement("Lagrange",cell,degree=self.degree,dim=2)

        element=MixedElement([element1,element2,element3])
        return element

    def dotv_coeffs(self):
        """   
        Set left hand side derivative coefficients
        """
#         return as_vector([0.,0.,0.,1000.,1000.])
        scale_p=1.
        scale_t=1.
        return as_vector([0.,0.,0.,1.*scale_p,1.*scale_t])

           
class CosseratTHM1DFEproblem(UserFEproblem):
    """
    Defines a user FE problem for given FE formulation
    """
    def __init__(self,FEformulation,PF):
        self.description="Example of 2D plane strain problem, Cosserat continuum with Drucker Prager material"
        scale = 1.
        self.nw=1
        self.problem_step=0
        self.Pressure_loading = PF
        self.Normal_loading_eff = -600./3*scale+self.Pressure_loading
        self.Normal_loading_total =self.Normal_loading_eff-self.Pressure_loading
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
        self.w=1.
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
            rreg=1.*self.w/(1.*np.float(self.nw))
            lreg=-1.*self.w/(1.*np.float(self.nw))
            return between(x[0], (lreg,rreg))

    class Gauss_point_Querry2(SubDomain):
        def __init__(self,w,nw):
            self.w=w
            self.nw=nw
            super().__init__()
            
        def inside(self, x, on_boundary):
            return between(x[0], (-self.w/2,self.w/2))


    def create_Gauss_point_querry_domain(self,mesh):
        """
        Create subdomains by marking regions
        """
        GaussDomain = MeshFunction("size_t", mesh, mesh.topology().dim())
        GaussDomain.set_all(0) #assigns material/props number 0 everywhere
        GaussDomainQuerry2= self.Gauss_point_Querry2(self.w,self.nw) #This takes all Gauss point along the line
        GaussDomainQuerry2.mark(GaussDomain,2)
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
        tmp[1-1]=self.Normal_loading_eff
        tmp[5-1]=self.Normal_loading_eff
        tmp[9-1]=self.Normal_loading_eff
        tmp[101-1]= self.Pressure_loading
        
        self.feobj.svars2.interpolate(Constant(tmp))
        
        #Modify the stresses (for Paraview)
        tmp=np.zeros(6)
        tmp[1-1]=self.Normal_loading_total

        self.feobj.sigma2.interpolate(Constant(tmp))

        tmp=np.zeros(5)
        tmp[4-1]= self.Pressure_loading 
    
        self.feobj.usol.interpolate(Constant(tmp))
                    
        pass
        
    def set_bcs(self):
        """
        Set boundary conditions for the user problem / could be replaced by external mesher, e.g. Abaqus, Gmsh...
        """

        scale_u=1./1000.
        self.u1_tot=0.1
        bcs=[]
        if self.problem_step == 0:
            bcs = [
            
                [1, [0, [0,0], 0.]],
                [1, [0, [0,1], 0.]],
                [1, [0, [1], 0.]],
                [1, [0, [2,0], 0.]],
                [1, [0, [2,1], 0.]],                
             
                [2, [3, [0,0], self.Normal_loading_total*scale_u]],
                # [2, [0, [0,0], 0.]],
                [2, [0, [0,1], 0.]],
                [2, [0, [1], 0.]],                
                [2, [0, [2,0], 0.]],
                [2, [0, [2,1], 0.]],                
                ]
        elif self.problem_step == 1:
            bcs = [
            
                [1, [0, [0,0], 0.]],
                [1, [0, [0,1], 0.]],
                [1, [0, [1], 0.]],
                [1, [0, [2,0], 0.]],
                [1, [0, [2,1], 0.]],                

                
                [2, [0, [0,0], -self.u1_tot/scale_u]],
                [2, [0, [0,1], 0.]],
                [2, [0, [1], 0.]],
                [2, [0, [2,0], 0.]],
                [2, [0, [2,1], 0.]],                
                   
                ]
        elif self.problem_step > 1:
            bcs = [
            
                [1, [0, [0,0], 0.]],
                [1, [0, [0,1], 0.]],
                [1, [0, [1], 0.]],
                [1, [0, [2,0], 0.]],
                [1, [0, [2,1], 0.]],                

                [2, [0, [0,0], -self.u1_tot/scale_u]],
                [2, [0, [0,1], 0.]],
                [2, [0, [1], 0.]], 
                [2, [0, [2,0], self.Pressure_loading]],
                [2, [0, [2,1], 0.]],                
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
              [2,[0,[1]]],
              [2,[1,[1]]],
              [2,[1,[2,0]]],
              [2,[0,[2,0]]],
              [2,[1,[2,1]]],
              [2,[0,[2,1]]],

              [3,[1,[0,0]]],
              [3,[0,[0,0]]],
              [3,[1,[0,1]]],
              [3,[0,[0,1]]],
              [3,[0,[1]]],
              [3,[1,[1]]],
              [3,[1,[2,0]]],
              [3,[0,[2,0]]],
              [3,[1,[2,1]]],
              [3,[0,[2,1]]],
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
                    
                    [1,[1,[9]]],  #mu_11
                    [1,[1,[10]]], #mu_12
                    [1,[1,[11]]], #mu_13
                    [1,[1,[12]]], #mu_21
                    [1,[1,[13]]], #mu_22
                    [1,[1,[14]]], #mu_23
                    [1,[1,[15]]], #mu_31
                    [1,[1,[16]]], #mu_32
                    [1,[1,[17]]], #mu_33

                    [1,[1,[18]]], #qhy_11
                    [1,[1,[19]]], #qhy_22
                    [1,[1,[20]]], #qhy_33
                    [1,[1,[21]]], #qth_11
                    [1,[1,[22]]], #qth_22
                    [1,[1,[23]]], #qth_33

                    
                    [1,[1,[24]]], #epsilon_11
                    [1,[1,[25]]], #epsilon_12
                    [1,[1,[26]]], #epsilon_13
                    [1,[1,[27]]], #epsilon_21
                    [1,[1,[28]]], #epsilon_22
                    [1,[1,[29]]], #epsilon_23
                    [1,[1,[30]]], #epsilon_31
                    [1,[1,[31]]], #epsilon_32
                    [1,[1,[32]]], #epsilon_33
                    [1,[1,[33]]], #kappa_11
                    [1,[1,[34]]], #kappa_12
                    [1,[1,[35]]], #kappa_13
                    [1,[1,[36]]], #kappa_21
                    [1,[1,[37]]], #kappa_22
                    [1,[1,[38]]], #kappa_23
                    [1,[1,[39]]], #kappa_31
                    [1,[1,[40]]], #kappa_32
                    [1,[1,[41]]], #kappa_33
                    
                    [1,[1,[42]]], #P,_11
                    [1,[1,[43]]], #P,_22
                    [1,[1,[44]]], #P,_33
                    [1,[1,[45]]], #T,_11
                    [1,[1,[46]]], #T,_22
                    [1,[1,[47]]], #T,_33
                                        
                    [1,[1,[75]]], #lambda_dot #2 plot over line
                    # [2,[1,[75]]], #lambda_dot #2 plot over line
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
        umat_lib = umat_lib_path+'/COSSERAT3D-THM/libplast_Cosserat3D-THM.so'
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

        g1=8./5.;g2=2./5.;g3=8./5.;g4=2./5.
        
        h1=2./3. ;h2=-1./6.;h3=2./3.;h4=-1./6.;
        
        K=666.66; G=1.*10.**3.; Gc=0.5*10.**3. ; L=1*10**(2.);R=10.*10.**(-3.);  
        MG=G*(R**2.)/h3 ; MGc=MG;
        permeability1 = 1.*10**-8;fluid_viscocity = 1.;bstar=1;
        permeability = permeability1/bstar
        conductivity = 1.*10**-8; rhoC = 1; alpha =0.; lstar = 0.*10.**8.;
        
        tanfi=0.5; cc=0.;
        tanpsi=0.; Hsfi=-0.; Hscc=-0.;
        eta1=0.0
        
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
        saveto=reference_data_path+"/HYDRO-PLASTIC/Cosserat_1D_Drucker-Prager_HP_test_step_0"+"_App_1_"+str(self.nw)+".xdmf"
        self.problem_step = 0
        self.bcs=self.set_bcs()
        self.feobj.symbolic_bcs = sorted(self.bcs, key=itemgetter(1))
        print("initial")
        converged=self.solve(saveto,summary=True)
        scale_t_program = [self.scale_t,self.scale_t,self.scale_t,self.scale_t,self.scale_t,self.scale_t]
        ninc=[100,100,100,100,100,100]
        print("shearing1")
    
        nsteps=2
        for i in range(nsteps):
            self.problem_step = i+1
            scale_t = scale_t_program[i]
            self.slv.nincmax=ninc[i]     
            self.slv.dtmax=0.1*scale_t
            self.slv.dt=self.slv.dtmax
            self.slv.tmax=self.slv.tmax+1.*scale_t
            self.feobj.symbolic_bcs = sorted(self.set_bcs(), key = itemgetter(1))
            self.feobj.initBCs()
            saveto= reference_data_path+"/HYDRO-PLASTIC/Cosserat_1D_Drucker-Prager_HP_test_step_"+str(i+1)+"_App_1_"+str(self.nw)+".xdmf"
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

        
    def svars_history_unpack(self,list1):
        for i,elem in enumerate(list1):
            if i==0:
                self.array_dtime=np.array([[elem[0]]])
                self.array_gp_svars_comp=elem[1].reshape((1,len(elem[1])))
                continue
            
            self.array_dtime=np.concatenate((self.array_dtime.copy(),np.array([[elem[0]]])))
            self.array_gp_svars_comp=np.concatenate((self.array_gp_svars_comp.copy(),elem[1].reshape((1,len(elem[1])))))
    
    def extract_generalized_force_disp(self):
        analysis_history=self.feobj.problem_history
        self.history_unpack(analysis_history)
        self.array_time=self.array_time[:].copy()
        self.array_gen_force=self.array_gen_force[:,:]
        self.array_gen_disp=self.array_gen_disp[:,:]
        return
    

    def extract_svars_gauss_point(self):
        analysis_svars_history=self.feobj.problem_svars_history
        self.svars_history_unpack(analysis_svars_history)
        self.array_dtime=self.array_dtime[:].copy()
        self.array_gp_svars_comp=self.array_gp_svars_comp[:,:].copy() 