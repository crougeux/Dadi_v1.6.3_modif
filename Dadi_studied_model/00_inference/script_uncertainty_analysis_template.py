#! ~/py33/bin
# -*- coding: utf-8 -*-

# Library importation
import os
import sys
import dadi
import numpy
import modeledemo_mis_new_models
import Godambe

# Global variables
# Best fit params:
p0 = [74.449901, 25.189265, 0.401979, 11.735954, 0.594012, 0.000064, 0.169386, 0.196292, 1.302698, 0.041998, 0.989006]
bootstrap_directory = "../0_1000_boot_fs/est/"
all_boot_file = os.path.join(bootstrap_directory, "all_boot.csv")
data_file = '../01-Data/est_26.fs'

# Load Custom Isolation withMigration with 2 Migration rates and exponential growth model: nu1, nu2, b1, b2, hrf, m12, m21, Ts, Tsc, Q, O
func = modeledemo_mis_new_models.SC2NG

# Make the extrapolating version of our demographic model function.
func_ex = dadi.Numerics.make_extrap_log_func(func)

# grid_pts is the set of grid points used in extrapolation
grid_pts = 26, 36, 46

# all_boot is a numpy array containing the boot-strapped data sets
all_boot = open(all_boot_file).readlines()
all_boot = [x.split(" ") for x in all_boot]
all_boot = numpy.array(all_boot)
all_boot = all_boot[0:10]

# data is the original data
data = dadi.Spectrum.from_file(data_file)

# mask singletons
data.mask[1, 0] = True
data.mask[0, 1] = True

# eps is the step size for derivative calculation
eps=0.01

# The final entry of the returned uncertainties will correspond to Theta
multinom = True

# uncert is a numpy array equal in length to p0
uncert = Godambe.GIM_uncert(func_ex, grid_pts, all_boot, p0, data, multinom, eps)

# Write to file
with open('SC2NG_WEB_GIM_uncert_eps0.01_masked.txt', 'w'):
    f.write("test\n")
    #f.write("p0 = [" + ", ".join(p0) + "]\nEstimated standard deviation of the model parameters:\n" + uncert)
