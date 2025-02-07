'''
Created on Aug 27, 2018

@author: Ioannis Stefanou

.. to do: test rollback
'''
import unittest
import numpy as np
from ngeoFE_unittests.Upscaling.SuperMaterials_Cauchy import Cauchy2DSuperFEmaterial

class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Load supermaterial
        '''
        cls.notfirsttime=True
        cls.my_supermaterial=Cauchy2DSuperFEmaterial()
        
    def test_init_exec(self):
        '''
        Tests initialization and execution
        '''
        svars=np.array([0.])
        GP_id=2
        stress=np.zeros(3); dsde=np.zeros(9); de=np.array([.1,.0,.0])
        self.my_supermaterial.usermatGP(stress, de, svars, dsde, 1., GP_id)
        print(stress,"\n")
        print(svars,"\n")
        print(de,'\n')
        print(dsde,'\n')
        values_diff=stress-np.array([.1,.0,.0])
        norm=np.linalg.norm(values_diff)
        values_diff=dsde-np.array([1.,0.,0.,0.,1.,0.,0.,0.,.5])
        norm+=np.linalg.norm(values_diff)
        #
        stress=np.zeros(3); dsde=np.zeros(9); de=np.array([.0,.1,.0])
        self.my_supermaterial.usermatGP(stress, de, svars, dsde, 1., GP_id)
        values_diff=stress-np.array([.1,.1,.0])
        norm=np.linalg.norm(values_diff)
        values_diff=dsde-np.array([1.,0.,0.,0.,1.,0.,0.,0.,.5])
        norm+=np.linalg.norm(values_diff)
        #
        stress=np.zeros(3); dsde=np.zeros(9); de=np.array([.0,.0,.1])
        self.my_supermaterial.usermatGP(stress, de, svars, dsde, 1., GP_id)
        values_diff=stress-np.array([.1,.1,.05])
        norm=np.linalg.norm(values_diff)
        values_diff=dsde-np.array([1.,0.,0.,0.,1.,0.,0.,0.,.5])
        norm+=np.linalg.norm(values_diff)
        self.assertTrue(norm<=1e-6, "Incorrect stresses or dsde "+str(norm))
        
    def test_exec_append(self):
        '''
        Tests execution and append for larger GP's
        '''
        svars=np.array([0.])
        GP_id=20
        stress=np.zeros(3); dsde=np.zeros(9); de=np.array([.1,.0,.0])
        self.my_supermaterial.usermatGP(stress, de, svars, dsde, 1., GP_id)
        values_diff=stress-np.array([.1,.0,.0])
        norm=np.linalg.norm(values_diff)
        values_diff=dsde-np.array([1.,0.,0.,0.,1.,0.,0.,0.,.5])
        norm+=np.linalg.norm(values_diff)
        #
        stress=np.zeros(3); dsde=np.zeros(9); de=np.array([.0,.1,.0])
        self.my_supermaterial.usermatGP(stress, de, svars, dsde, 1., GP_id)
        values_diff=stress-np.array([.1,.1,.0])
        norm=np.linalg.norm(values_diff)
        values_diff=dsde-np.array([1.,0.,0.,0.,1.,0.,0.,0.,.5])
        norm+=np.linalg.norm(values_diff)
        #
        stress=np.zeros(3); dsde=np.zeros(9); de=np.array([.0,.0,.1])
        self.my_supermaterial.usermatGP(stress, de, svars, dsde, 1., GP_id)
        values_diff=stress-np.array([.1,.1,.05])
        norm=np.linalg.norm(values_diff)
        values_diff=dsde-np.array([1.,0.,0.,0.,1.,0.,0.,0.,.5])
        norm+=np.linalg.norm(values_diff)
        self.assertTrue(norm<=1e-6, "Incorrect stresses or dsde "+str(norm))
    
        
if __name__ == '__main__':
    unittest.main()

