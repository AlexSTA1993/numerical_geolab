"""
Created on Aug 1, 2018

@author: Ioannis Stefanou


Contains various incremental solvers.

.. todo:: 
    * Implement central finite difference scheme
    * Implement forward Euler
    * Implement Crank-Nicolson
    * Implement Newmark scheme
    * Implement arc-length
"""
import tracemalloc
from dolfin import *

import numpy as np
from math import *
from ngeoFE.ngio import Msg, FileExporter

from dolfin.cpp.la import LUSolver

#from dolfin.cpp.common import MPI_barrier
# from dolfin.cpp.common import MPI_sum
from dolfin.cpp.la import LUSolver, KrylovSolver

from dolfin import MPI as mpi

import cProfile
import pstats
import io

class Backward_Euler_Solver():
    """
    Newton-Raphson Solver for time dependent and time independent problems (backward Euler method)

    :param feobj: finite element object
    :type feobj: FEobject 
    :param mats: list of material objects
    :type mats: UserMaterial
    """
    def __init__(self, feobj, mats):
        """
        Initialize default parameters for the solver
        """
        self.t=0.
        self.tmax=1.
        self.dinc=1. # cut in dinc increments e.g. 2.
        self.dtmax=self.tmax/self.dinc
        self.dtmin=1e-8 #1e-8
        self.nincmax=500 # max increments; if more stops
        self.incmodulo=-1 # output to file every incmodulo increments
        self.inctime=-1 # output to file every inctime duration of increments. 
        #
        self.waitpatience=5 # set alwsays to >0 or to infinity to desactivate
        self.reducedtfactor=2. # factor to reduce time step if problem. Same factor used for increase.
        #
        self.nitermax=100 # max iterations; if more reduces dt
        #
        self.convergence_tol=1e-6 # necessary tolerance for residual
        self.eps=DOLFIN_EPS # accuracy
        self.feobj=feobj
        self.mats=mats
        self.set_init_stress=True
        self.assembleDCpreservingsymmetry=False
        self.removezerolines=True
        # Define solver and solver options
        #LSsolver = KrylovSolver("gmres", "ilu")
        #LSsolver.parameters["verbose"] = True
        #for item in LSsolver.parameters.items():
        #    print(item)
        self.LSsolver=LUSolver() #10/3/2019

#         self.LSsolver=PETScLUSolver()

#         self.LSsolver=KrylovSolver("bicgstab", "hypre_euclid")
#         self.LSsolver=KrylovSolver("gmres", "ilu")
#         self.LSsolver=KrylovSolver("gmres", "hypre_euclid")
#         self.LSsolver=KrylovSolver("cg", "hypre_amg")

        #self.LSsolver.parameters['absolute_tolerance'] = 1e-12
        #self.LSsolver.parameters['relative_tolerance'] = 1e-7

    def set_initial_stress(self,sts,dt=0.):
        '''
        Sets initial stresses

        :param dt: Time increment (normally unnecessary)
        :type dt: double
        :param sts: Messaging object
        :type sts: Msg

        :ivar feobj.u0: initial converged displacement field contained in the finite element object
        :ivar feobj.deGP2: converged strain as projected from the initial displacement field
        :ivar deGP: strain increment
        :ivar aux_deGP: strain increment of auxilary fields
        :ivar stress_t: current value of the problem stress tensor in Voight form
        :ivar svars_t: current value of the material state variables
        :ivar dsde_tdt: current value of the problem stiffness tensor
        :ivar nill: value indicating that the material algorithm has converged (nill=0)
        :ivar sigma2: updated value of the problem stress tensor
        :ivar svars2: updated value of the material state variables
        :ivar dsde2: updated values of the problem stiffness tensor

        :return nill: value indicating that the material algorithm has converged (nill=0)
        :rtype nill: integer
        '''
        self.feobj.local_project(self.feobj.epsilon2(self.feobj.u0), self.feobj.Vstress, self.feobj.deGP2)
        deGP=np.reshape(self.feobj.deGP2.vector().get_local(),(-1,self.feobj.p_nstr))
        aux_deGP=np.reshape(self.feobj.aux_deGP2.vector().get_local(),(-1,self.feobj.p_aux))
        stress_t=np.reshape(self.feobj.sigma2.vector().get_local(),(-1,self.feobj.p_nstr))
        svars_t=np.reshape(self.feobj.svars2.vector().get_local(),(-1,self.feobj.p_nsvars))
        dsde_tdt=np.reshape(self.feobj.dsde2.vector().get_local(),(-1,self.feobj.p_nstr**2))

        nill=self.umat(stress_t,deGP,aux_deGP,svars_t,dsde_tdt,self.feobj.domainidGP,dt)
        if nill==1:
            sts.PrintMsg("material problem; have to reduce increment @ init stress","red")
        else:
            self.feobj.sigma2.vector().set_local(np.reshape(stress_t,-1))
            self.feobj.svars2.vector().set_local(np.reshape(svars_t,-1))
            self.feobj.dsde2.vector().set_local(np.reshape(dsde_tdt,-1))
            #print("hello20",svars_t)
        return nill

    def solve(self,outputfile="",silent=False,summary=True,start_dtmin=False):#,first_step=True):
        """
        Solves the problem with backward Euler

        :param outputfile: output hdmf filename for saving results
        :param silent: messages display
        :param summary: display incrementation summary 
        :type outputfile: string
        :type silent: boolean
        :type summary: boolean
        :return: True if converged
        :rtype: boolean 
        """
        sfs=FileExporter(self.feobj,outputfile)
        self.out=sfs
        sts=Msg()
        sts.silent=silent
        waitbeforeincreasedt=0
        t=self.t
        tinit=self.t
        tfinal=self.tmax
        told=self.t
        tolds=told
        if start_dtmin == True:
            dt=self.dtmin #starts with the minimum icrement and increases gradually
        else:
            dt=self.dtmax
        ninc=1
        nill=0
        #
        #Initialize Boundary Conditions
        self.feobj.initBCs()
        #Initialize Jacobian and Residual
        self.feobj.init_Jac_res()
        #
        usol_nodal=None; b_nodal=None
        # Save externally - output
        sfs.export(t, self.feobj)
        # Set initial stresses
        if self.t==0. or self.set_init_stress==True:
            nill=self.set_initial_stress(sts,dt=0.)
            nnill=mpi.sum(self.feobj.comm,nill)
            mpi.barrier(self.feobj.comm)
            if nnill!=0:
                converged=False
                if self.feobj.comm.Get_rank()==0: sts.PrintMsg("Failed to set initial stresses - quitting", "red") 
                return converged
            else:
                self.set_init_stress=False
                converged=True
        if t==self.tmax:
            converged=True
            return converged
        # Increment
        while t<self.tmax: # and ninc<=nincmax:

            t=min(told+dt,self.tmax); dt=t-told; self.feobj.set_dt(dt)
            if self.feobj.comm.Get_rank()==0: sts.PrintMsg("Increment: "+str(ninc)+" Time "+str(t)+" *** DTime "+str(dt))
            # Set boundary conditions 
            self.feobj.incrementBCs(t,told,tinit,tfinal)    
            # Set volumic forces
            self.feobj.setfi(dt) 

            self.feobj.Du.interpolate(Constant(np.zeros(self.feobj.ndofs)))
            self.feobj.local_project(self.feobj.epsilon2(self.feobj.Du), self.feobj.Vstress, self.feobj.deGP2)
            self.feobj.local_project(self.feobj.aux_field2(self.feobj.Du), self.feobj.Vaux, self.feobj.aux_deGP2)
            #
            deGP=np.reshape(self.feobj.deGP2.vector().get_local(),(-1,self.feobj.p_nstr))
            aux_deGP=np.reshape(self.feobj.aux_deGP2.vector().get_local(),(-1,self.feobj.p_aux))
            stress_t=np.reshape(self.feobj.sigma2.vector().get_local(),(-1,self.feobj.p_nstr))
            svars_t=np.reshape(self.feobj.svars2.vector().get_local(),(-1,self.feobj.p_nsvars))
            dsde_t=np.reshape(self.feobj.dsde2.vector().get_local(),(-1,self.feobj.p_nstr**2))
            #
            svars_tdt=np.copy(svars_t)
            dsde_tdt=np.copy(dsde_t)   

            if self.assembleDCpreservingsymmetry==False:
                # Assemble without preserving symmetry        
                A=assemble(self.feobj.Jac); b=assemble(self.feobj.Res)
                # save nodal value for history output
                if self.feobj.history_indices_ti!=None: b_nodal=b.get_local()[self.feobj.history_indices_ti] #ALEX 13/05/2022
#                 print(1,b.get_local())
                for bc in self.feobj.DCbcs: bc.BC.apply(A,b)
            else:              
                abcs=[bc.BC for bc in self.feobj.DCbcs]
                # Assemle preserving symmetry
                A,b=assemble_system(self.feobj.Jac,self.feobj.Res,abcs)
                # save nodal value for history output
                if self.feobj.history_indices_ti!=None: b_nodal=b.get_local()[self.feobj.history_indices_ti] #ALEX 13/05/2022
            if self.removezerolines==True: A.ident_zeros()
            nRes=b.norm("l2")
            if summary==False:
                if self.feobj.comm.Get_rank()==0: sts.PrintMsg("   Iteration: "+str(ninc)+".0    Residual:"+str(nRes))
            niter=0
            while nRes > self.convergence_tol or nill==1 or niter==0: 
                #
                self.LSsolver.solve(A, self.feobj.du.vector(), b)#,"mumps")
                #solve(A, du.vector(), b,"gmres", "ilu") #"lu")#,"mumps")
                self.feobj.Du.assign(self.feobj.Du+self.feobj.du)
                ndu=self.feobj.du.vector().norm("l2")
                if isnan(ndu):                
                    if self.feobj.comm.Get_rank()==0: sts.PrintMsg("Displacement increment is not a number; have to reduce increment", "red")
                    nill=1
                    
                    self.feobj.sigma2.vector().set_local(np.reshape(stress_t,-1))
                    self.feobj.dsde2.vector().set_local(np.reshape(dsde_t,-1))
                    dt=dt/self.reducedtfactor
                    break 

                if self.feobj.large_displacements == True:
                    self.feobj.update_mesh(self.feobj.mesh,self.feobj.du)
                if self.feobj.comm.Get_rank()==0: sts.PrintMsg("    |du|: "+str(ndu))
                # Calculate total strain increments at Gauss points
                self.feobj.local_project(self.feobj.epsilon2(self.feobj.Du), self.feobj.Vstress, self.feobj.deGP2)
                deGP=np.reshape(self.feobj.deGP2.vector().get_local(),(-1,self.feobj.p_nstr))
                # Calculate auxiliary total strain increments at Gauss points
                self.feobj.local_project(self.feobj.aux_field2(self.feobj.Du), self.feobj.Vaux, self.feobj.aux_deGP2)
                aux_deGP=np.reshape(self.feobj.aux_deGP2.vector().get_local(),(-1,self.feobj.p_aux))
                #
                stress_tdt=np.copy(stress_t)
                svars_tdt=np.copy(svars_t)
                #
                nill=self.umat(stress_tdt,deGP,aux_deGP,svars_tdt,dsde_tdt,self.feobj.domainidGP,dt)
                if nill==1: sts.PrintMsg("   Process "+str(self.feobj.comm.Get_rank())+": material problem; have to reduce increment - 1", "red")
                # Assure synchronous execution in case of material problem
                nnill=mpi.sum(self.feobj.comm,nill)
                mpi.barrier(self.feobj.comm)
                if nnill!=0: nill=1
                if nill==1:
                    if self.feobj.large_displacements == True:
                        self.feobj.update_mesh(self.feobj.mesh,self.feobj.Du,minus=True)

                    self.feobj.sigma2.vector().set_local(np.reshape(stress_t,-1))
                    self.feobj.dsde2.vector().set_local(np.reshape(dsde_t,-1))
                    dt=dt/self.reducedtfactor

                    del stress_tdt
                    del svars_tdt

                    break
                elif niter==self.nitermax:
                    if self.feobj.large_displacements == True:
                        self.feobj.update_mesh(self.feobj.mesh,self.feobj.Du,minus=True)

                    if self.feobj.comm.Get_rank()==0: sts.PrintMsg("   reached max iterations; reducing increment - 3", "red")
                    self.feobj.sigma2.vector().set_local(np.reshape(stress_t,-1))
                    self.feobj.dsde2.vector().set_local(np.reshape(dsde_t,-1))
                    dt=dt/self.reducedtfactor

                    del stress_tdt
                    del svars_tdt
#                     del dsde_tdt

                    break
                else: 
                    self.feobj.sigma2.vector().set_local(np.reshape(stress_tdt,-1))
                    self.feobj.dsde2.vector().set_local(np.reshape(dsde_tdt,-1))
                    # When all done, assemble
                    if self.assembleDCpreservingsymmetry==False:
                        # Assemble without preserving symmetry   
                        A=assemble(self.feobj.Jac); b=assemble(self.feobj.Res)
                        # save nodal value for history output
                        if self.feobj.history_indices_ti!=None: b_nodal=b.get_local()[self.feobj.history_indices_ti]
                        for bc in self.feobj.DCbcs0: bc.BC.apply(A,b)
                    else:
                        abcs=[bc.BC for bc in self.feobj.DCbcs0]
                        # Assemle preserving symmetry
                        A,b=assemble_system(self.feobj.Jac,self.feobj.Res,abcs)              
                        # save nodal value for history output
                        if self.feobj.history_indices_ti!=None: b_nodal=b.get_local()[self.feobj.history_indices_ti]  
                    if self.removezerolines==True: A.ident_zeros()

                    del stress_tdt

                    nRes=b.norm("l2")
                    niter += 1
                    if summary==False:
                        if self.feobj.comm.Get_rank()==0: sts.PrintMsg("    Residual: "+str(nRes)+"\t    |du|: "+str(ndu))

            if isnan(nRes):                
                if self.feobj.comm.Get_rank()==0: sts.PrintMsg("Failed to converge, residual is not a number - stopping", "red")
                converged=False
                break
            elif (dt<self.dtmin and nRes > self.convergence_tol) or (dt<self.dtmin and nill==1):
                if self.feobj.comm.Get_rank()==0: sts.PrintMsg("Failed to converge, reached dtmin - stopping")
                converged=False
                break
            elif nill==1: 
                waitbeforeincreasedt=0
                converged=False
                t=told
            elif niter==self.nitermax and nRes > self.convergence_tol:
                waitbeforeincreasedt=0
                converged=False
                t=told
            elif ninc==self.nincmax:
                if self.feobj.comm.Get_rank()==0: sts.PrintMsg("Failed to converge, reached max increments - stopping")
                converged=False
                break
            elif nill==0:
                # Increment converged
                if waitbeforeincreasedt==self.waitpatience:
                    dtnew=min(dt*self.reducedtfactor,self.dtmax,self.tmax-told)
                    if (dtnew-dt)/dt>self.eps:
                        if self.feobj.comm.Get_rank()==0: sts.PrintMsg("   Good behavior noticed, increasing timestep from " + str(dt) + " to " + str(dtnew), "green")

                        dt=dtnew
                    waitbeforeincreasedt=0
                waitbeforeincreasedt+=1   
                # Update state variables 
                self.feobj.svars2.vector().set_local(np.reshape(svars_tdt,-1))

                # Update solution 
                self.feobj.usol.assign(self.feobj.usol+self.feobj.Du)
                # Print summary
                if summary==True:
                    if self.feobj.comm.Get_rank()==0: sts.PrintMsg("   Iterations: "+str(niter)+"    Residual: "+str(nRes))
                # Save externally - output
                if ((t-tolds)>=self.inctime and ninc%self.incmodulo<=0) or t==self.tmax:
                    tolds=t
                    sfs.export(t, self.feobj)
                #update time


                told=t
                self.t=t
                ninc+=1
                converged=True

                # save nodal values for history output

                del svars_tdt
                del dsde_tdt
                if self.feobj.history_indices_ui!=None: 
                    usol_nodal=self.feobj.usol.vector().get_local()[self.feobj.history_indices_ui]
                    svars_gauss_points=self.feobj.svars2.vector().get_local()[self.feobj.svars_history_indices]
                    self.feobj.problem_history.append([t,b_nodal,usol_nodal])
                    self.feobj.problem_svars_history.append([dt,svars_gauss_points])

        return converged

    def umat(self,stress_t,deGP,aux_deGP,svars_t,dsde_t,domainidGP,dt):
        """
        Calls user material at Gauss points

        :param stress_t: generalized stress - input/output
        :param deGP: generalized deformation vector - input
        :param aux_deGP: auxiliary generalized deformation vector - input
        :param svars_t: state variables - input/output
        :param dsde_t: jacobian - output
        :param domainidGP: domain id for choosing the right material
        :param dt: time increment
        :type stress_t: numpy array
        :type deGP: numpy array
        :type aux_deGP: numpy array
        :type svars_t: numpy array
        :type dsde_t: numpy array
        :type domainidGP: numpy array
        :type dt: double
        :return nill: parameter indicating convergence of material
        :rtype nill: integer
        """
        for GP_id in range(int(deGP.size/self.feobj.p_nstr)):
            nill = self.mats[domainidGP[GP_id]].usermatGP(stress_t[GP_id],deGP[GP_id],svars_t[GP_id],dsde_t[GP_id],dt,GP_id,aux_deGP[GP_id])
            if nill==1:
                return nill
        return 0
