'''
Created on Dec 4, 2019

@author: alexandrosstathas
'''
import unittest
from ngeoFE_unittests.Multiphysics.Cosserat_tests.OneD.BVP.test1D_Thermal_Elastic import THM1D_FEformulation, THM1D_FEproblem
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
print(os.getcwd())
reference_data_path = ngeo_parameters.reference_data_path    
reference_data = reference_data_path+'analytical.dat' 
temperature_list = [] 
with open(reference_data, newline='') as f:
    reader = csv.reader(f,delimiter=' ')
    for row in reader:
        row = row[0].split(',')
#         print(row)
        temperature_list.append(float(row[1]))
#         tenperature_list = np.array(temperature_list)
# print(len(temperature_list))
# print(temperature_list) 
# print(temperature_list)
reference_data_path = ngeo_parameters.reference_data_path    
reference_data = reference_data_path+'temperature_transient_values.out'  
f = open(reference_data,'wb')
pickle.dump(temperature_list,f)
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
        saveto=reference_data_path+"./test1D_Thermoelasticity.xdmf"
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
      
      
    def test_temperature_transient(self):
        values=self.my_FEproblem.temperature
        
        reference_data_path = ngeo_parameters.reference_data_path    
        reference_data = reference_data_path+'temperature_transient_values.out'    
        with open(reference_data, "rb") as fp:   # Unpickling
            values_ref = pickle.load(fp)
        
        values_ref=np.array(values_ref) 
        values_diff=values_ref-values
#         from matplotlib import pyplot as plt  
        time=np.linspace(0,1,num=51)
        plt.plot(time,values,'-bo' , markersize = 2,label = 'FE_values')
        plt.plot(time,values_ref, '-', color='orange',marker = "+", markersize = 8, label = 'Mathematica_values')
#         ax1.set(xlabel='time (s)', ylabel='Temperature $C$',title='Temperature versus time at free end')
        plt.grid()
        plt.legend(loc = 'lower right')
        plt.show()
        equal=abs(np.linalg.norm(values_diff))<=1.e-13
        self.assertTrue(equal, "Not identical temperature: "+str(abs(np.linalg.norm(values_diff))))
          
if __name__ == '__main__':
    unittest.main()