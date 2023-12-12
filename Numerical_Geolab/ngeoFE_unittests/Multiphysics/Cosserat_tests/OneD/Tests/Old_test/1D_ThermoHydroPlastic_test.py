'''
Created on Dec 6, 2019

@author: alexandrosstathas
'''

import unittest
from ngeoFE_unittests.Multiphysics.Cosserat_tests.OneD.BVP.test1D_Thermo_Hydro_Plastic import THM1D_FEformulation, THM1D_FEproblem
from ngeoFE_unittests import ngeo_parameters
# ngeo_parameters.reference_data_path='/home/alexandrosstathas/eclipse-workspace/numerical_geolab/Numerical_Geolab/ngeoFE_unittests/Multiphysics/reference_data/'


from matplotlib import pyplot as plt    
from dolfin import *
# from dolfin.cpp.io import HDF5File
# from dolfin.cpp.mesh import MeshFunction
#
import pickle
import os
import numpy as np
import csv
print('Im here')
print(os.getcwd())
reference_data_path = ngeo_parameters.reference_data_path    
reference_data = reference_data_path+'analytical.dat' 
print(reference_data)
pressure_list = [] 
with open(reference_data, newline='') as f:
    reader = csv.reader(f,delimiter=' ')
    for row in reader:
        row = row[0].split(',')
#         print(row)
        pressure_list.append(float(row[1]))
#         tenperature_list = np.array(temperature_list)
# print(len(temperature_list))
# print(temperature_list) 
# print(temperature_list)
reference_data_path = ngeo_parameters.reference_data_path    
reference_data = reference_data_path+'temperature_transient_values.out'
print(reference_data)  
f = open(reference_data,'wb')
pickle.dump(pressure_list,f)
f.close()
 
class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Run FE analysis example
        '''
        cls.notfirsttime=True
        cls.my_FEformulation=THM1D_FEformulation()
        cls.my_FEproblem=THM1D_FEproblem(cls.my_FEformulation)
        cls.my_FEproblem.give_me_solver_params()
        saveto=reference_data_path+"./test1D_ThermoHydroPlastic.xdmf"
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
        self.assertTrue(equal1, "Not identical normal stresses: "+str(abs(np.linalg.norm(values_diff1))))
        self.assertTrue(equal2, "Not identical normal stresses2: "+str(abs(np.linalg.norm(values_diff2))))
        self.assertTrue(equal3, "Not identical shear stresses: "+str(abs(np.linalg.norm(values_diff3))))
        self.assertTrue(equal4, "Not correct shear stress: "+str(abs(np.linalg.norm(values_diff4))))

    def test_pressure_vs_temperature_vs_volumetric_strain(self):
        values_diff=self.my_FEproblem.pressure_vs_temperature_vs_volumetric_strain()
        equal=abs(np.linalg.norm(values_diff))<=1.e-10
        self.assertTrue(equal, "Not identical temperature: "+str(abs(np.linalg.norm(values_diff))))
          
if __name__ == '__main__':
    unittest.main()