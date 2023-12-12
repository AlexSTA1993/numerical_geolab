'''
Created on Nov 5, 2018

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cauchy continua in 1D linear elasticity

'''
import unittest
from ngeoFE_unittests.Mechanics.Cauchy.TwoD.BVP.Cauchy2D_Von_Mises import Cauchy2DFEformulation, Cauchy2DFEproblem

from dolfin import *

from dolfin.cpp.io import HDF5File

import pickle
import numpy as np

from ngeoFE_unittests import ngeo_parameters
from ngeoFE_unittests import plotting_params 

# ngeo_parameters.reference_data_path='/home/alexandrosstathas/eclipse-workspace/numerical_geolab/Numerical_Geolab/ngeoFE_unittests/Mechanics/reference_data/'

reference_data_path = ngeo_parameters.reference_data_path    

class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Run FE analysis example
        '''
        cls.notfirsttime=True
        cls.my_FEformulation=Cauchy2DFEformulation()
        cls.my_FEproblem=Cauchy2DFEproblem(cls.my_FEformulation)
        cls.my_FEproblem.give_me_solver_params()
        cls.converged=cls.my_FEproblem.run_analysis_procedure(reference_data_path)
        
        if cls.converged==True: cls.my_FEproblem.plot_me()
        
        # dsde_data=cls.my_FEproblem.feobj.dsde2.vector().get_local().reshape((-1,1))
        # dsde_values=open(ngeo_parameters.reference_data_path+"dsde_values_VM.text","w")
        #
        # for row in dsde_data:
        #     np.savetxt(dsde_values,row)
        #
        # dsde_values.close()
        #
        # print(cls.my_FEproblem.feobj.dsde2.vector().get_local().shape)    
        
    def test_execution(self):
        '''
        Tests execution and convergence
        '''
        self.assertTrue(self.converged, "Convergence failed")
    
    def test_shear_force_displacement_values(self):
        '''
        Tests calculated nodal forces and displacements to values in ./reference_data/Cauchy2D_force_disp_values.out
        '''
        self.my_FEproblem.extract_force_disp()
        values_time = self.my_FEproblem.array_time
        values_force = self.my_FEproblem.array_force
        values_disp = self.my_FEproblem.array_disp
        # print(values_time.shape, values_disp.shape, values_force.shape)
        values=np.concatenate((values_time, values_disp, values_force), axis=1)
        # print(values.shape)
        # write data to binary files
        #with open(reference_data_path+"/Cauchy2D_Von_Mises_force_disp_values.out", "wb") as fp:   #Pickling
        #    pickle.dump(values, fp)
        
        #read data from binary files
        with open(reference_data_path+"/Cauchy2D_Von_Mises_force_disp_values.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff=values_ref-values

        equal=abs(np.linalg.norm(values_diff))<=1.e-9 #docker precision is reduced it however better than the accuracy of the equlibrium solver 10e-6 and material solver 10e-8
        self.assertTrue(equal, "Not identical time, displacements, forces: "+str(abs(np.linalg.norm(values_diff))))
        #asserts that data are correct
        #if dtat are correct then plot diagram
        if equal:
            x=list(values_disp[1:].copy())
            y=list(-values_force[1:].copy())
            x.insert(0,0)
            y.insert(0,0)
    
            # fig, ax = plotting_params.object_plot_axes('$u$ [mm]', '$\sigma$ [kPa]', '')
            # plotting_params.object_plot(x, y, ax, '')
            # plotting_params.plot_legends('./reference_data/', fig, ax,legend_title=' ', filename='Cauchy_2D_elastoplastic',mode='1')

            fig, ax1, color1 = plotting_params.object_plot_axes('$u$ [mm]', y1_txt='$\sigma$ [MPa]',color1='',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x, y,' y2', ax1, 'ax2', mode='1',color1=color1,color2='',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Von_Mises_elastoplasticity_sigma_u',mode='1')

    
    def test_identical_elastoplastic_matrix(self):  
        self.my_FEproblem.extract_force_disp()  
        self.my_FEproblem.extract_elastoplastic_matrix()  
        values=self.my_FEproblem.EH
       
        #write data to binary files
        # with open(reference_data_path+"/Cauchy2D_Von_Mises__elastoplastic_modulo.out", "wb") as fp:   #Pickling
        #     pickle.dump(values, fp)  
        

                #read data from binary files
        with open(reference_data_path+"/Cauchy2D_Von_Mises__elastoplastic_modulo.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff=values_ref-values

        equal=abs(np.linalg.norm(values_diff))<=1.e-13
        self.assertTrue(equal, "Not identical elastoplastic_moduli: "+str(abs(np.linalg.norm(values_diff))))    
    
    def test_analytical_elastoplastic_matrix(self):  
        self.my_FEproblem.extract_force_disp()  
        self.my_FEproblem.extract_elastoplastic_matrix()  
        values=self.my_FEproblem.EH

        with open(reference_data_path+"/Cauchy2D_Von_Mises__elastoplastic_modulo.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
   
        values_diff=values_ref[-1]+(100./1.1)
        # print(values_diff,values_ref[-1],-(100./1.1))
        equal=abs(np.linalg.norm(values_diff))<=1.e-8
        self.assertTrue(equal, "Not identical_analytical_moduli: "+str(abs(np.linalg.norm(values_diff))))        

        
if __name__ == '__main__':
    unittest.main()
