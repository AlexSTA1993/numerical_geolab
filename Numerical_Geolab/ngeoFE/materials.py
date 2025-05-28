'''
Created on Aug 3, 2018, updated on Feb 4, 2025

@author: Ioannis Stefanou & Nicholas Collins-Craft
'''
from ctypes import CDLL, byref, POINTER, RTLD_GLOBAL, c_double, c_int
import numpy as np
import juliacall


class UserMaterial():
    """
    Material class with a user material subroutine that is called from a shared library (e.g. a .so file)
    """
    def __init__(self, env_lib, umat_lib, umat_id):
        """
        Load libraries

        :param env_lib: environment libraries filenames with path
        :type env_lib: List of strings
        :param umat_lib: material library filename with path
        :type umat_lib: string
        :param umat_id: material id
        :type umat_id: integer
        """
        # Load external libraries
        for lib1 in env_lib:
            CDLL(lib1, mode=RTLD_GLOBAL)
        self.umatlib = CDLL(umat_lib)
        self.c_fvar_p = POINTER(c_double)  # Set a pointer type (called once and used later)
        self.umat_id = umat_id
        self.props = []

    def usermatGP(self, stressGP_t, deGP, svarsGP_t, dsdeGP_t, dt, GP_id, aux_deGP=np.zeros(1)):
        """
        User material at a Gauss point with a material subroutine that is called from a shared library

        :param stressGP_t: generalized stress at GP - input/output
        :type stressGP_t: numpy array
        :param deGP: generalized deformation vector at GP - input
        :type deGP: numpy array
        :param aux_deGP: auxiliary generalized deformation vector at GP - input
        :type aux_deGP: numpy array
        :param svarsGP_t: state variables at GP - input/output
        :type svarsGP_t: numpy array
        :param dsdeGP_t: jacobian at GP - output
        :type dsde_t: numpy array
        :param dt: time increment
        :type dt: double
        :param GP_id: Gauss Point id (global numbering of all Gauss Points in the problem) - for normal materials is of no use
        :type GP_id: integer
        :return: 0 if ok, 1 if failed
        :rtype: integer
        """
        # Create ctype variables
        __nstr = stressGP_t.size
        __NSTRGP = c_int(__nstr)
        __AUX_DEGP = c_int(aux_deGP.size)
        __NSVARSGP = c_int(svarsGP_t.size)
        __NPROPS = c_int(len(self.props))
        __NILL = c_int(0)
        __UMATID = c_int(self.umat_id)
        __DTIME = c_double(dt)
        # Get pointers for all
        __PROPS_p = self.props.ctypes.data_as(self.c_fvar_p)
        __DE_p = deGP.ctypes.data_as(self.c_fvar_p)
        __AUX_DE_p = aux_deGP.ctypes.data_as(self.c_fvar_p)
        __STRESS_p = stressGP_t.ctypes.data_as(self.c_fvar_p)
        __DSDE_p = dsdeGP_t.ctypes.data_as(self.c_fvar_p)
        __SVARS_p = svarsGP_t.ctypes.data_as(self.c_fvar_p)
        # Call the material library
        self.umatlib.usermaterial_(
            byref(__UMATID), 
            __STRESS_p, 
            __DE_p, 
            __AUX_DE_p, 
            __DSDE_p, 
            byref(__NSTRGP), 
            byref(__AUX_DEGP), 
            __PROPS_p, 
            byref(__NPROPS), 
            __SVARS_p, 
            byref(__NSVARSGP), 
            byref(__DTIME), 
            byref(__NILL)
            )     
        return __NILL.value


class UserJuliaMaterial():
    """
    Material class with a user material subroutine that is called from a Julia file
    """
    def __init__(self, env_lib, umat_lib, umat_parameters, umat_id):
        """
        Load Julia material. Env lib is a dummy variable for compatibility with the previous class
        :param env_lib: environment libraries filenames with path
        :type env_lib: List of strings
        :param umat_lib: material library filename with path
        :type umat_lib: string
        :param umat_id: material id
        :type umat_id: integer
        """
        # First read in the umat library
        with open(umat_lib, encoding="utf-8") as f:
            data = f.readlines()
            # Write the umat parameters to the first line of the file
            data[0] = "include(\"" + umat_parameters + "\")\n"
        # Write the modified data back to the file
        with open(umat_lib, 'w', encoding="utf-8") as f:
            f.writelines(data)
        # Load the material library
        umat_lib_string = 'include("' + umat_lib + '")'
        jl = juliacall.newmodule(umat_lib)
        self.jl = jl  # Store the Julia module
        self.jl.seval(umat_lib_string)
        # Convert the umat_id to a Julia variable
        jl_umat_id = jl.convert(jl.Int64, umat_id)
        self.umat_id = jl_umat_id
        self.props = []

    def usermatGP(self, stressGP_t, deGP, svarsGP_t, dsdeGP_t, dt, GP_id, aux_deGP=np.zeros(1)):
        """
        User material at a Gauss point with a material subroutine that is called from a Julia file

        :param stressGP_t: generalized stress at GP - input/output
        :type stressGP_t: numpy array
        :param deGP: generalized deformation vector at GP - input
        :type deGP: numpy array
        :param aux_deGP: auxiliary generalized deformation vector at GP - input
        :type aux_deGP: numpy array
        :param svarsGP_t: state variables at GP - input/output
        :type svarsGP_t: numpy array
        :param dsdeGP_t: jacobian at GP - output
        :type dsde_t: numpy array
        :param dt: time increment
        :type dt: double
        :param GP_id: Gauss Point id (global numbering of all Gauss Points in the problem) - for normal materials is of no use
        :type GP_id: integer
        :return: 0 if ok, 1 if failed
        :rtype: integer
        """

        # Convert the type flag to a Julia variable
        C_type_flag = self.jl.convert(self.jl.Int, 0)

        # Call the material library
        C_type_flag = self.jl.seval('usermaterial_')(self.umat_id, stressGP_t, deGP, svarsGP_t, dsdeGP_t, dt, GP_id, aux_deGP, C_type_flag)
        __NILL = c_int(C_type_flag)
        return __NILL.value
