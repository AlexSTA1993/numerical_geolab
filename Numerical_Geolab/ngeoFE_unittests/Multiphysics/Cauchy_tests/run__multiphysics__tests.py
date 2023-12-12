'''
Created on May 5, 2022

@author: Alexandros Stathas

run specific Multiphysics unittests.
'''
import unittest

def test_all():
    
    testmodules = [
'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_DP_Hydroelasticity_tests_0',
'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_DP_Hydroplasticity_tests_0',
'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_DP_Hydroplasticity_tests_1',
'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_DP_Thermo_Hydro_plasticity_tests_1',
'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_Thermo_Hydro_plasticity_tests_0',
'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_Thermoelasticity_tests_0',
'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_ThermoHydroelasticity_tests',
'ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.Tests.Cauchy3D_Thermoplasticity_tests_0'
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