'''
Created on Dec 4, 2019

@author: alexandrosstathas
'''
# import sys
# print(sys.path)
# sys.path.insert(0, "/home/alexandrosstathas/eclipse-workspace/numerical_geolab/Numerical_Geolab")

import unittest
from ngeoFE_unittests.Multiphysics.Cosserat_tests.OneD.BVP.test1D_Thermo_Hydro_Plastic_Rattez import THM1D_FEformulation, THM1D_FEproblem
from ngeoFE_unittests import ngeo_parameters
# ngeo_parameters.reference_data_path='/home/alexandrosstathas/eclipse-workspace/numerical_geolab/Numerical_Geolab/ngeoFE_unittests/Multiphysics/reference_data/'


from matplotlib import pyplot as plt    
# from dolfin import *
import numpy as np
import os
# from dolfin.cpp.io import HDF5File
# from dolfin.cpp.mesh import MeshFunction
#

 
class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Run FE analysis example
        '''
        print(os.getcwd())
        cls.notfirsttime=True
        cls.my_FEformulation=THM1D_FEformulation()
        cls.my_FEproblem=THM1D_FEproblem(cls.my_FEformulation)
        cls.my_FEproblem.give_me_solver_params()
        reference_data_path = ngeo_parameters.reference_data_path  
#         print('SOSOSOS',reference_data_path) 
        saveto=reference_data_path+"./test1D_Hydro_Plastic_Rattez1.xdmf"
        cls.converged=cls.my_FEproblem.solve(saveto,silent=True)
        if cls.converged==True: cls.my_FEproblem.plot_me()
#         print("analysis finished")
#
#         with open("pf_static_values.out", "wb") as fp:   #Pickling
#              pickle.dump(values, fp)
    def test_execution(self):
        '''
        Tests execution and convergence
        '''
        self.assertTrue(self.converged, "Convergence failed")
    def test_yield_criterion_particular_case(self):
        values_diff, values_diff1, values_diff2, values_diff3, values_diff4 =self.my_FEproblem.yield_criterion_particular_case()
        equal=abs(np.linalg.norm(values_diff))<=1.e-10
        equal1=abs(np.linalg.norm(values_diff1))<=1.e-10
        equal2=abs(np.linalg.norm(values_diff2))<=1.e-10
        equal3=abs(np.linalg.norm(values_diff3))<=1.e-10
        equal4=abs(np.linalg.norm(values_diff4))<=1.e-10
        self.assertTrue(equal, "Yield criterion breach: "+str(abs(np.linalg.norm(values_diff))))
#         self.assertTrue(equal1, "Not identical normal stresses: "+str(abs(np.linalg.norm(values_diff1))))
        self.assertTrue(equal2, "Not identical normal stresses2: "+str(abs(np.linalg.norm(values_diff2))))
        self.assertTrue(equal3, "Not identical shear stresses: "+str(abs(np.linalg.norm(values_diff3))))
#         self.assertTrue(equal4, "Not correct shear stress: "+str(abs(np.linalg.norm(values_diff4))))
    
#     def test_plastic_work_vs_dtemp(self):
#         values_diff=self.my_FEproblem.dtemp_vs_plastic_work()
#         equal=abs(np.linalg.norm(values_diff))<=1.e-10
#         self.assertTrue(equal, "Plastic work not identical to dtemp: "+str(abs(np.linalg.norm(values_diff))))
    
    def test_pressure_vs_temperature_vs_volumetric_strain(self):
        values_diff=self.my_FEproblem.pressure_vs_temperature_vs_volumetric_strain()
        equal=abs(np.linalg.norm(values_diff))<=1.e-10
        self.assertTrue(equal, "Not identical temperature: "+str(abs(np.linalg.norm(values_diff))))
          
if __name__ == '__main__':
    unittest.main()