'''
Created on Oct 29, 2018

@author: Ioannis Stefanou
'''

import unittest

def test_all():
    
    testmodules = [
        
        #Mechanics
        ## Cauchy
#        'ngeoFE_unittests.Mechanics.Cauchy.TwoD.Tests.Cauchy2D_Drucker_Prager_tests_App_1',
#        'ngeoFE_unittests.Mechanics.Cauchy.TwoD.Tests.Cauchy2D_Von_Mises_Perzyna_tests_App_1',
#        'ngeoFE_unittests.Mechanics.Cauchy.TwoD.Tests.Cauchy2D_Von_Mises_Perzyna_tests_App_2',#(long test)
#        'ngeoFE_unittests.Mechanics.Cauchy.TwoD.Tests.Cauchy2D_Von_Mises_Perzyna_tests_App_3_1',
#        'ngeoFE_unittests.Mechanics.Cauchy.TwoD.Tests.Cauchy2D_Von_Mises_Perzyna_tests_App_3',
        
        # #Mechanics
        # ## Cosserat
        #'ngeoFE_unittests.Mechanics.Cosserat.OneD.Tests.Cosserat1D_Drucker_Prager_tests_App_1',
#        'ngeoFE_unittests.Mechanics.Cosserat.OneD.Tests.Cosserat1D_Drucker_Prager_tests_App_2',#(long test)
        #
        # #Multiphysiscs
        # ##Cauchy
#        'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_DP_Hydroelasticity_tests_0',
#        'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_DP_Hydroplasticity_tests_0',
#        'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_DP_Hydroplasticity_tests_1',
#        'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_DP_Thermo_Hydro_plasticity_tests_1',
#        'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_Thermo_Hydro_plasticity_tests_0',
#        'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_Thermoelasticity_tests_0',
#        'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_ThermoHydroelasticity_tests', #(long test)
#        'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_Thermoplasticity_tests_0',
        #
        # #Multiphysiscs
        # ##Cosserat
        'ngeoFE_unittests.Multiphysics.Cosserat_tests.OneD.Tests.Cosserat1D_DP_Hydroelasticity_tests_0',
        'ngeoFE_unittests.Multiphysics.Cosserat_tests.OneD.Tests.Cosserat1D_DP_Hydroplasticity_tests_0',
        'ngeoFE_unittests.Multiphysics.Cosserat_tests.OneD.Tests.Cosserat1D_DP_Thermoelasticity_tests_0',
        'ngeoFE_unittests.Multiphysics.Cosserat_tests.OneD.Tests.Cosserat1D_Thermo_Hydro_plasticity_tests_0',

#       # #Materials
#        'ngeoFE_unittests.Materials.DruckerPrager_tests',
#        'ngeoFE_unittests.Materials.Viscoplastic_DP_tests',
#        'ngeoFE_unittests.Upscaling.SuperMaterials_tests',
#        'ngeoFE_unittests.Materials.Asym_CamClay_tests',
#        'ngeoFE_unittests.Upscaling.SuperMaterials_tests',
        ]
    
    suite = unittest.TestSuite()
    
    for t in testmodules:
        try:
            # If the module defines a suite() function, call it to get the suite.
            mod = __import__(t, globals(), locals(), ['suite'])
            suitefn = getattr(mod, 'suite')
            suite.addTest(suitefn())
        except (ImportError, AttributeError):
            # else, just load all the test cases from the module.
            suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))
    
    unittest.TextTestRunner().run(suite)


if __name__ == '__main__':
    test_all()
