'''
Created on Aug 2, 2018

@author: Ioannis Stefanou
'''

import sys
import csv
from dolfin import *
from dolfin.cpp.io import XDMFFile


# Define a class to print messages with colours during simulation
class Msg():
    """
    Print messages with colours
    """
    def __init__(self):
        """
        Set colour codes
        """
        self.colours = {"RED": "\033[1;31m",
                       "BLUE": "\033[1;34m",
                       "CYAN": "033[1;36m",
                       "GREEN": "\033[0;32m",
                       "RESET": "\033[0;0m",
                       "BOLD": "\033[;1m",
                       "REVERSE": "\033[;7m"}
        self.silent = False

    def PrintMsg(self, message, colour="RESET", replace=False):
        """
        Print message in colour and reset to default

        :param message: message to display
        :type message: string
        :param colour: style to use (RED,BLUE,CYAN,GREEN,RESET,BOLD,REVERSE)
        :type colour: string
        :param replace: replace text in same line
        :type replace: boolean
        """
        if self.silent == True:
            return
        sys.stdout.write(self.colours[colour.upper()])
        if replace == False:
            print(message)
        else:
            print(message, end="\r")
        sys.stdout.write(self.colours["RESET"])


class FileExporter():
    """
    Export solution data to file

    :param feobj: finite element object
    :type feobj: FEobject 
    :param file: xdmf filename to save data
    :type file: string
    """
    def __init__(self, feobj, file=""):
        """
        Set necessary spaces and filenames for outout and plotting
        """
        if file != "":
            self.output = True
            __Vtmp = VectorElement("DG", feobj.cell, degree=0, dim=feobj.p_nstr)  # P_NSTR components
            self.SIGMAavgFS = FunctionSpace(feobj.mesh, __Vtmp)
            self.SIGMAavg = Function(self.SIGMAavgFS, name="Stresses")
            __Vtmp = VectorElement("DG", feobj.cell, degree=0, dim=feobj.p_nsvars)  # P_NSVARS components
            self.SVARSavgFS = FunctionSpace(feobj.mesh, __Vtmp)
            self.SVARSavg = Function(self.SVARSavgFS, name="State_Variables")
            self.xdmffile = XDMFFile(feobj.comm,file)           
            self.xdmffile.parameters["flush_output"] = True
            self.xdmffile.parameters["functions_share_mesh"] = True
            self.xdmffile.parameters['rewrite_function_mesh'] = False
        else:
            self.output = False
    
    def export(self, t, feobj):
        """
        Output results

        :param t: time
        :type t: double
        :param feobj: finite element object
        :type feobj: FEobject 
        """
        if self.output == True:
            self.xdmffile.write(feobj.usol, t)
            self.SIGMAavg.vector().set_local(project(feobj.sigma2, self.SIGMAavgFS).vector().get_local())
            self.xdmffile.write(self.SIGMAavg, t)
            self.SVARSavg.vector().set_local(project(feobj.svars2, self.SVARSavgFS).vector().get_local())
            self.xdmffile.write(self.SVARSavg, t)
            return 0
        else:
            return 1


def export_list_to_csv(filename,list):
    with open(filename, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(list)
    return



