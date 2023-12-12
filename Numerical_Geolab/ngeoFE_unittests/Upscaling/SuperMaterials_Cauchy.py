'''
Created on Aug 27, 2018

@author: Ioannis Stefanou
'''
from ngeoFE.superFEmaterials import SuperFEMaterial
from ngeoFE.feproblem import General_FEproblem_properties
from ngeoFE.materials import UserMaterial
#
import numpy as np
import time
#
from dolfin import *
from ufl.operators import Dx
from ufl.tensors import as_vector
from ufl.finiteelement.mixedelement import VectorElement

from ngeoFE_unittests import ngeo_parameters

reference_data_path = ngeo_parameters.reference_data_path   

class Cauchy2DSuperFEmaterial(SuperFEMaterial):
    def __init__(self):
        super().__init__("Cauchy_2D")
    
    class SuperFEMaterialFEformulation(SuperFEMaterial.SuperFEMaterialFEformulation):
        '''
        Defines a user FE formulation for the supermaterial
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
                Dx(v[0],0),             #gamma_11
                Dx(v[1],1),             #gamma_11
                Dx(v[0],1)+Dx(v[1],0)   #gamma_12
                ] 
            return as_vector(gde)
        
        def create_element(self,cell):
            """
            Set desired element
            """
            # Defines a Lagrangian FE of degree 1 for the displacements
            element_disp=VectorElement("Lagrange",cell,degree=1,dim=2)
            return element_disp
#             
    class SuperFEMaterialFEproblem(SuperFEMaterial.SuperFEMaterialFEproblem):
        """
        Defines FE problem for given FE formulation of the supermaterial
        
        Only orthogonal elementary cells are supported
        """
        def set_general_properties(self):
            """
            Set here all the parameters of the problem, except material properties 
            
            When overidden, has to define the maximum number of the state variables of the materials of the supermaterial by setting: self.genprops.p_nsvars= ??
            """
            self.genprops=General_FEproblem_properties()
            # Number of state variables
            self.genprops.p_nsvars=27
            # Periodic cell corners 
            self.left_bottom_corner=[0.,0.]
            self.right_top_corner=[1.,1.]
        
        def create_mesh(self):
            """
            Set mesh and subdomains
             
            :return: mesh object
            :rtype: Mesh
            """
            # Generate mesh
            mesh=UnitSquareMesh(20,20)
            cd=MeshFunction("size_t", mesh, mesh.topology().dim())
            fd=MeshFunction("size_t", mesh, mesh.topology().dim()-1)
            return mesh,cd,fd
        
        def set_materials(self):
            """
            Create material objects and set material parameters
            
            :return: Returns a list of UserMaterial objects
            :rtype: UserMaterial
            """
            mats=[]
            
            env_lib=ngeo_parameters.env_lib        #umat_lib='./libplast_Cauchy3D-DP.so'
            umat_lib_path= ngeo_parameters.umat_lib_path
            umat_lib = umat_lib_path+'/CAUCHY3D-DP/libplast_Cauchy3D-DP.so'
            # load material #1
            umat_id=2       # if many materials exist in the same library
            mat=UserMaterial(env_lib,umat_lib,umat_id)
            mat.props=self.set_material_1_properties(1.,0.,1000.)
            #
            mats.append(mat)
            # load material #2
            umat_id=2       # if many materials exist in the same library
            mat=UserMaterial(env_lib,umat_lib,umat_id)
            mat.props=self.set_material_1_properties(1.,0.,1000.)
            #
            mats.append(mat)
            return mats
        
        def set_material_1_properties(self,EE,nu,cc):
            """
            Sets material parameters
            """
            GG=EE/(2.*(1.+nu))
            KK=EE*GG/(3.*(3.*GG-EE))
            props=np.array([KK,GG,0.,0.,0.,0.,0.,0.,0.,0.,0.,cc,0.,0.,0.,0.,0.,0.,0.,0.])
            props=props.astype("double")
            return props