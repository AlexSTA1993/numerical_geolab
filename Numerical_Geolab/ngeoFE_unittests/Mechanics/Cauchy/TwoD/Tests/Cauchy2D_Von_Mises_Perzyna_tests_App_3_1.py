'''
Created on Nov 5, 2018

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cauchy continua in 1D linear elasticity

'''
import sys
import os
import unittest
from ngeoFE_unittests.Mechanics.Cauchy.TwoD.BVP.Cauchy2D_Von_Mises_Perzyna_App_3_1 import Cauchy2DFEformulation, Cauchy2DFEproblem

import pickle
import numpy as np

from ngeoFE_unittests import ngeo_parameters
from ngeoFE_unittests import plotting_params 

reference_data_path = ngeo_parameters.reference_data_path    

# Check if the environment variable or command-line argument is set to activate plots
activate_plots = False

if 'RUN_TESTS_WITH_PLOTS' in os.environ and os.environ['RUN_TESTS_WITH_PLOTS'].lower() == 'true':
    activate_plots = True
elif len(sys.argv) > 1 and sys.argv[1].lower() == 'with_plots':
    activate_plots = True

class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Run FE analysis example
        '''
        cls.notfirsttime=True
        cls.my_FEformulation=Cauchy2DFEformulation()
        
        #first slow loading procedurde
        imperfection1=0.1
        cls.my_FEproblem1=Cauchy2DFEproblem(cls.my_FEformulation,imperfection1)
        cls.my_FEproblem1.give_me_solver_params(scale_t=10e-9)
        cls.converged1=cls.my_FEproblem1.run_analysis_procedure(reference_data_path)       
        
        imperfection2=0.05
        cls.my_FEproblem2=Cauchy2DFEproblem(cls.my_FEformulation,imperfection2)
        cls.my_FEproblem2.give_me_solver_params(scale_t=10e-9)
        cls.converged2=cls.my_FEproblem2.run_analysis_procedure(reference_data_path)       
        
    def test_execution(self):
        '''
        Tests execution and convergence
        '''
        self.assertTrue(self.converged1, "Convergence failed")
    
    def test_shear_force_displacement_values(self):
        '''
        Tests calculated nodal forces and displacements to values in ./reference_data/Cauchy2D_force_disp_values.out
        '''
        self.my_FEproblem1.extract_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_force1 = self.my_FEproblem1.array_force
        values_disp1 = self.my_FEproblem1.array_disp
        
        self.my_FEproblem2.extract_force_disp()
        values_time2 = self.my_FEproblem2.array_time
        values_force2 = self.my_FEproblem2.array_force
        values_disp2 = self.my_FEproblem2.array_disp
        
        values1=np.concatenate((values_time1, values_disp1, values_force1), axis=1)
        
        # with open(reference_data_path+"Cauchy2D_Perzyna_force_disp_values_App_3_1.out", "wb") as fp:   #Pickling
        #     pickle.dump(values1,fp)        
        
        with open(reference_data_path+"Cauchy2D_Perzyna_force_disp_values_App_3_1.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff=values_ref-values1
    
        equal=abs(np.linalg.norm(values_diff))<=1.e-9 #docker precision is reduced it however better than the accuracy of the equlibrium solver 10e-6 and material solver 10e-8
        self.assertTrue(equal, "Not identical time, displacements, forces: "+str(abs(np.linalg.norm(values_diff))))
        #asserts that data are correct
        #if data are correct then plot diagram
        if equal and activate_plots:
            x1=list(values_time1[:].copy())
            y1=list(-values_force1[:].copy())
            x1.insert(0,0)
            y1.insert(0,0)

            x2=list(values_time2[:].copy())
            y2=list(-values_force2[:].copy())
            x2.insert(0,0)
            y2.insert(0,0)
            
            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$\sigma$ [kPa]',color1='k',y2_txt='$\sigma$ [kPa]',color2='', title='',mode='1')
            
            plotting_params.object_plot(x1, y1,y2, ax1, 'ax2',x2, mode='3',color1='r',color2='b',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Perzyna_visoplasticity_App_3_1_sigma_t',mode='1')

            x1=list(values_disp1[:].copy())
            y1=list(-values_force1[:].copy())
            x1.insert(0,0)
            y1.insert(0,0)

            x2=list(values_disp2[:].copy())
            y2=list(-values_force2[:].copy())
            x2.insert(0,0)
            y2.insert(0,0)

            fig, ax1, color1 = plotting_params.object_plot_axes('$u$ [mm]', y1_txt='$\sigma$ [kPa]',color1='k',y2_txt='$\sigma$ [kPa]',color2='', title='',mode='1')

            plotting_params.object_plot(x1, y1,y2, ax1, "ax2",x2, mode='3',color1='g',color2='c',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Perzyna_visoplasticity_App_3_1_sigma_u',mode='1')


    def test_localization_width_values(self):
        '''
        localization width is equal to the imperfection
        '''
        self.my_FEproblem1.extract_svars_gauss_point()
        self.my_FEproblem2.extract_svars_gauss_point()

        nf1=0+2*self.my_FEproblem1.ny#-2
        nf2=0+2*self.my_FEproblem2.ny#-2

        ldot_values_over_line_1 = self.my_FEproblem1.array_gp_svars_comp[:,0:nf1]
        ldot_values_over_line_2 = self.my_FEproblem2.array_gp_svars_comp[:,0:nf2]

        x_coord_values_overline_1 = np.array(self.my_FEproblem1.feobj.svars_coordinates[0:nf1])[:,0].flatten()
        x_coord_values_overline_2 = np.array(self.my_FEproblem2.feobj.svars_coordinates[0:nf2])[:,0].flatten()
        
        x1=x_coord_values_overline_1.tolist()
        y1=ldot_values_over_line_1[-1,:].flatten()
        
        x2=x_coord_values_overline_2.tolist()
        y2=ldot_values_over_line_2[-1,:].flatten()
        
        if activate_plots==True:
        
            fig, ax1, ax2, color1, color2 = plotting_params.object_plot_axes('$h [mm]$', y1_txt='$\dot{\lambda}_{0.2}$',color1='r',y2_txt='$\dot{\lambda}_{0.1}$',color2='b', title='',mode='2')
            plotting_params.object_plot(x1, y1, y2, ax1, ax2, x2, mode='2',color1='r',color2='b',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_2D_viscoplastic_App_3_1_x_ldot',mode='1')
        
if __name__ == '__main__':
    unittest.main()
