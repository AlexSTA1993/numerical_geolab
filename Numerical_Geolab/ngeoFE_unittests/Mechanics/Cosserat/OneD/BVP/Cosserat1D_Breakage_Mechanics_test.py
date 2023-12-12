from dolfin import *
import time
import numpy as np
from ngeoFE.feproblem import UserFEproblem, General_FEproblem_properties
from ngeoFE.fedefinitions import FEformulation
from ngeoFE.materials import UserMaterial
from ngeoFE_unittests import ngeo_parameters
from ngeoFE_unittests import plotting_params
import os  # allows easier manipulation of directories
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from dolfin.cpp.io import HDF5File
from operator import itemgetter
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

 # Set the material values
K = 13833.  # Bulk stiffness
G = 7588.  # Shear stiffness
zeta = 1.  # Elastic-to-plastic Cosserat length scale parameter
h1 = 6./5.  # Cosserat stress invariant weighting factor (with 3D kinematic model)
h2 = 3./10.  # As above
h3 = 6./5.*zeta**2  # As above
h4 = 3./10.*zeta**2  # As above
Gc = 3*G/(2*(h1 - h2))  # The Cosserat shear stiffness (fully determined by our other parameters)
L = 0.00001  # The Cosserat torsional stiffness (zero in the model, given a very small positive value for numerical reasons)
H = 3*G/(2*(h3 + h4))  # The first Cosserat bending stiffness (fully determined by our other parameters)
Hc = 3*G/(2*(h3 - h4))  # The second Cosserat bending stiffness (fully determined by our other parameters)
theta_gamma = 0.80  # The grain size distribution parameter accompanying the strains
theta_kappa = 0.89  # The grain size distribution parameter accompanying the stresses
Ec = 4.65  # The critical grain crushing energy
M = 1.7  # The slope of the critical state line in :math:`p-q` space (mean stress - triaxial deviatoric stress)
omega = 70.  # Coupling angle describing the system's tendency to crush grains or favour pore collapse. It is given here in degrees (we will later automatically convert it to radians)
x_r = 0.105  # The reference grain size
beta = 1.  # A legacy parameter that is unimportant for understanding the model

# Set the initial value of B
B_nought = 0.0
# Target confinement stress as a fraction of p_crit
p_frac = 0.3
# Set the target gamma (i.e. the equivalent homogeneous strain we wish to subject the system to)
gamma = 0.2

# Set the system size (in mm)
h = 17.5
# Set the rescaling factor. A value of 1 solves the system "as written". Values larger than this (say 1000) cause the system to be solved more quickly, at the cost of some accuracy.
rescale_factor = 1
# Set the desired tolerance level (for this code, material will have +1) for the residual. This can also be changed to increase accuracy at the cost of speed (or vice versa), provided the material has been compiled at the requested tolerance level.
tolerance_level = 5
# Set the number of elements that we want
element_number = 641

class Cosserat1DFEformulation(FEformulation):
    '''
    Defines a user FE formulation
    '''
    def __init__(self):
        # Number of stress/deformation components
        self.p_nstr = 4
        # Number of Gauss points
        self.ns = 2

    def generalised_epsilon(self, v):
        """
        Set user's generalised deformation vector
        """
        gde = [
            Dx(v[0], 0)/rescale_factor,  # gamma_11
            v[2]/rescale_factor,  # gamma_12
            (Dx(v[1], 0) - v[2])/rescale_factor,  # gamma_21
            Dx(v[2], 0)/rescale_factor  # kappa_31
            ]
        return as_vector(gde)

    def create_element(self, cell):
        """
        Set desired element
        """
        # Defines a Lagrangian FE of degree 2 for the displacements
        element_disp = VectorElement("Lagrange", cell, degree=2, dim=1)
        # Defines a Lagrangian FE of degree 1 for the rotations
        element_rot = FiniteElement("Lagrange", cell, degree=1, dim=1)
        # Creates a mixed element for Cosserat medium
        element = MixedElement([element_disp, element_rot])
        return element

class bottom(SubDomain):
 def inside(self, x, on_boundary):
     return x[0] < 0.  and on_boundary

class top(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0. and on_boundary


class CosseratBKG1DFEproblem(UserFEproblem):
    """
    Defines a user FE problem for given FE formulation
    """
    def __init__(self, FEformulation):
        self.description = "Example of a 1D shearing problem in the Cosserat continuum"
        self.problem_step = 0
        self.h = h
        super().__init__(FEformulation)

    def set_general_properties(self):
        """
        Set here all the parameters of the problem, except material properties
        """
        self.genprops = General_FEproblem_properties()
        # Number of state variables
        self.genprops.p_nsvars = 76
        
    def create_mesh(self):
        """
        Set mesh and subdomains
        """
        # Generate mesh
        ny = 641
        h = self.h
        mesh = IntervalMesh(ny, -h, h)
        cd = MeshFunction("size_t", mesh, mesh.topology().dim())
        fd = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        return mesh, cd, fd
    
    def create_subdomains(self, mesh):
        """
        Create subdomains by marking regions
        """
        subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
        subdomains.set_all(0) # assigns material/props number 0 everywhere
        return subdomains
    
    def mark_boundaries(self, boundaries):
        """
        Mark bottom and top boundary points
        """
        boundaries.set_all(0)
        top0 = top()
        top0.mark(boundaries, 1)
        bottom0 = bottom()
        bottom0.mark(boundaries, 2)
        return

    def set_bcs(self):
        """
        Set boundary conditions for the user problem. These take the structure
        [region_id, [bc_type, [dof], value]], where region_id is the label we gave
        the boundary in the mark_boundaries() function (1 or 2 in this case),
        bc_type is one of Dirichlet (0), Neumann (1) or Robin (2), dof is the degree
        of freedom we apply the boundary condition to (we are not obliged to specify
        a condition on every dof on the boundary if we don't wish to), and value
        is the value we set the boundary condition to take
        """
        h = self.h
        p_crit_zero = np.sqrt(2*K*Ec/theta_gamma)
        stress_target = p_frac*p_crit_zero
        strain_target = stress_target/K
        u_n = strain_target*2*h*rescale_factor
        u_t = gamma*2*h*rescale_factor
        if self.problem_step == 0:
            bcs = [
                [1, [0, [0, 0], 0.]],
                [1, [0, [0, 1], 0.]],
                [1, [0, [1], 0.]],
                [2, [0, [0, 0], u_n]],
                [2, [0, [0, 1], 0.]],
                [2, [0, [1], 0.]]
                ]
        elif self.problem_step == 1:
            bcs = [
                [1, [0, [0, 0], 0.]],
                [1, [0, [0, 1], 0.]],
                [1, [0, [1], 0.]],
                [2, [2, [0, 0], u_n]],
                [2, [0, [0, 1], u_t]],
                [2, [0, [1], 0.]]
                ]
        return bcs

    def set_materials(self):
        """
        Create material objects and set material parameters
        """
        mats = []
        # load material #1
        env_lib = ngeo_parameters.env_lib
        umat_lib_path = ngeo_parameters.umat_lib_path
        umat_lib = umat_lib_path + 'COSSERAT3D-BREAKAGE/libplast_Cosserat3D-Breakage.so'
        umat_id = 1       # if many materials exist in the same library
        mat = UserMaterial(env_lib, umat_lib, umat_id)
        mat.props = self.set_material_properties()
        mats.append(mat)
        return mats

    def set_material_1_properties(self):
        """
        Sets material parameters
        """
        omega_rad = omega*math.pi/180.  # We convert the coupling angle to radians
        c2wEc = math.cos(omega_rad)**2/Ec  # We find cos^2(omega)/Ec (for numerical convenience)
        s2wEc = math.sin(omega_rad)**2/Ec  # We find sin^2(omega)/Ec (for numerical convenience)
        props = np.array([K, G, Gc, L, H, Hc, theta_gamma, theta_kappa, beta, x_r, Ec, M, c2wEc, s2wEc, h1, h2, h3, h4, 0.])  # Load the properties as an array
        props = props.astype("double")
        return props


path1 = '../reference_data'
my_FEformulation = Cosserat1DFEformulation()
my_FEproblem = CosseratBKG1DFEproblem(my_FEformulation)
saveto = path1 + "c_rescaled_" + str(rescale_factor) + ".xdmf"
my_FEproblem.slv.dtmax = .1
converged = my_FEproblem.solve(saveto)  # This saves the first confining loading part of the problem
# Now we change the boundary conditions
my_FEproblem.problem_step = 1
my_FEproblem.bcs = my_FEproblem.set_bcs()
my_FEproblem.feobj.symbolic_bcs = sorted(my_FEproblem.bcs, key=itemgetter(1))

my_FEproblem.slv.tmax = 2.
my_FEproblem.slv.dtmax = .5
my_FEproblem.slv.nitermax = 500
my_FEproblem.slv.nincmax = 1000000
my_FEproblem.slv.convergence_tol = 1.*10**(-tolerance_level)  # has to be bigger than the materials

saveto = path1 + "s_rescaled_" + str(rescale_factor) + ".xdmf"
converged = my_FEproblem.solve(saveto, summary=False)
