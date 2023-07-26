# Finds the electron density of the other atoms using Edwards/Flory-Huggins excluded volume
# and Fermi-Amaldi self-interaction instead of LDAX exchange functional.
# Done in a spherical box of radius rc with a spectral code. 
# Using atomic units.

from numpy import zeros, pi, sqrt, array, diag, transpose, matmul, exp
from numpy import sum, linspace, sin, arange, trapz, log, ones, savez, load, where
from numpy import append, roll, einsum, size, max, min, float64
from numpy.linalg import eigh, solve
from pylab import plot, show
from findgamma import findgamma
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import time

start = time.time()

# The shell_config array represents the number of electrons in each shell. This is dynamic
# DISCLAIMER: Make sure the length of shell_config is the 
# same as the number of shells you will need for your final element
# For example, if you require 7 shells then set shell_config for Hydrogen as
# shell_config = [1,0,0,0,0,0,0]
# Goal for C   = [2,2,1,1]
# Lines for the graphs

# These variables are for line colors of plots and naming files/plots:
total_line = 'b-'
shell_lines = ['k--','g-','r--','c:','m-','y--','b:'] # For plots: Need to add enough colours in this array to match the shell config! These lines are for graphing!
elements = ["Hydrogen", "Helium", "Lithium", "Beryllium", "Boron", "Carbon", "Nitrogen", "Oxygen", "Fluorine", 
"Neon", "Sodium", "Magnesium", "Aluminium", "Silicon","Phosphorus","Sulfur","Chlorine","Argon"]
element_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
max_ele = len(elements)
experiment = "_finding_g0inv" # Name your experiment, files will be saved with this string suffixed to it

# Helper Functions Start vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def angstroms(bohr_radius): # Converts radii from Bohr units to Angstroms
    return 0.52918 * bohr_radius

def sign(num): # Outputs the sign (positive = +1 or negative = -1 or zero = 0) of a number
    if num < 0:
        return -1
    elif num > 0:
        return 1
    else:
        return 0

def consolidate(arr_of_arr): # Takes a list of arrays and consolidates row-wise (along the first axis) it into a 1D array 
        rows = size(arr_of_arr,0)
        cons = []
        for i in range(0,rows):
            cons = np.append(cons,arr_of_arr[i])
        return cons # Example: [[1,2,3],[4,5,6],[7,8,9]] -> [1,2,3,4,5,6,7,8,9]
# Helper Functions End ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Initialize g0inv, beta, nb and gamma values
# g0inv = 5.742468000
g0inv = 1421.752468

# Column Labels for data to be saved to a pandas dataframe
column_labels_g0inv = {'g0inv': []#,
                    #    'Radii': [],
                    #    'Differences': []
                    }		
column_labels_elements = {"Hydrogen": [], 
                        "Helium": [], 
                        "Lithium": [], 
                        "Beryllium": [],
                        "Boron": [], 
                        "Carbon": [], 
                        "Nitrogen": [], 
                        "Oxygen": [], 
                        "Fluorine": [], 
                        "Neon": [], 
                        "Sodium": [], 
                        "Magnesium": [], 
                        "Aluminium": [], 
                        "Silicon": [],
                        "Phosphorus": [],
                        "Sulfur": [],
                        "Chlorine": [],
                        "Argon": []}
column_labels_differences = {'He - H': [],
                            'Li - He': [],
                            'Be - Li': [],
                            'B - Be': [],
                            'C - B': [],
                            'N - C': [],                            
                            'O - N': [],
                            'F - O': [],
                            'Ne - F': [],
                            'Na - Ne': [],
                            'Mg - Na': [],
                            'Al - Mg': [],
                            'Si - Al': [],
                            'P - Si': [],
                            'S - P': [],
                            'Cl - S': [],
                            'Ar - Cl': []
                            }

# Initialize pandas dataframes
g0inv_data = pd.DataFrame(column_labels_g0inv)	
radii_data = pd.DataFrame(column_labels_elements)		
differences_data = pd.DataFrame(column_labels_differences)	

# Set beta values for each element
beta_arr = [20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            17,
            14,
            12,
            10,
            9,
            8,
            7]
nb = 400 # Number of basis functions. 
gamma = findgamma(nb) # Pack gamma from fortran.

Ashcroft_trend = [-0.20, # Difference in radii of consecutive atoms according to Ashcroft
                   0.86,
                  -0.01,
                  -0.14,
                  -0.15,
                  -0.11,
                  -0.08,
                  -0.08,
                  -0.07,
                   0.69,
                   0.15,
                  -0.01,
                  -0.07,
                  -0.09,
                  -0.09,
                  -0.08,
                  -0.09]

element_configs = [[1,0,0], # H
                   [2,0,0], # He
                   [2,1,0], # Li
                   [2,2,0], # Be
                   [2,3,0], # B
                   [2,4,0], # C
                   [2,5,0], # N
                   [2,6,0], # O
                   [2,7,0], # F
                   [2,8,0], # Ne
                   [2,8,1], # Na
                   [2,8,2], # Mg
                   [2,8,3], # Al
                   [2,8,4], # Si
                   [2,8,5], # P
                   [2,8,6], # S
                   [2,8,7], # Cl
                   [2,8,8]] # Ar

consecutive_trend_errors = 0 # Keeps track of how many times the trend is not followed
trend = True # If trend is not followed, then change g0inv
while True: # Break condition for this loop is at the very end of the program
    print("======================================================================")
    print("======================================================================")
    radii = zeros(max_ele, float) # Array to store element radii
    differences = zeros(max_ele-1, float) # Array to store difference in element radii
    # radii = pd.Series(zeros(16), dtype=float64) # Series to store element radii
    # differences = pd.Series(zeros(15), dtype=float64) # Array to store difference in element radii
    if(not(trend)):
        g0inv += 1
        if(g0inv <= 0):
            break
        print("g0inv has mutated to", g0inv)

    for shell_config in element_configs: # Configuration of electrons
        shells       = len(shell_config) # Number of shells
        num          = sum(shell_config) # Total number of electrons in the atom.

        # These variables are for naming files and plots:
        cur_ele_num     = num-1
        prev_ele_num    = num-2
        two_ele_ago_num = num-3

        current_element  = elements[cur_ele_num]  # Calculate the current element using the electron number
        previous_element = elements[prev_ele_num] # Calculate the previous element using the electron number
        two_elements_ago = elements[two_ele_ago_num]
        print("Running calculations for", current_element)

        ################################COMMENTS##########################################
        # Pros of this program: No need to change 16 different variables every time you add a shell
        # Cons of this program: It's somewhat harder to access a specific variable for a specific shell
        # Documentation of created arrays for troubleshooting:
        # FORMAT:  arr = [old_variables]            (shape)          data types
        #       nb_arr = [nb,2*nb,3*nb,...]         (shells)         array of ints
        # shell_config = [num1,num2,num3,...]       (shells)         array of ints
        #           wb = [wb1,wb2,wb3,...]          (shells, nb)     array of arrays of floats
        #   shell_dens = [dens1,dens2,dens3,...]    (shells, nb)     array of arrays of floats
        #    realn_arr = [realn1,realn2,realn3,...] (shells, nrr)    array of array of floats
        #    dens0_arr = [dens01,dens02,dens03,...] (shells)         array of floats
        #           wp = [wp1,wp2,wp3,...]          (shells, nb)     array of arrays of floats
        #         Amat = [Amat1,Amat2,Amat3,...]    (shells, nb, nb) array of nb x nb matricies 
        #          val = [val1,val2,val3,...]       (shells, nb)     array of arrays of eigenvalues
        #          vec = [vec1,vec2,vec3,...]       (shells, nb, nb) array of arrays of eigenvectors
        #         vect = [vect1,vec2,vec3,...]      (shells, nb, nb) array of arrays of transposed eigenvectors
        #         Dval = [Dval1,Dval2,Dval3,...]    (shells, nb, nb) array of nb x nb matricies
        #           qp = [qp1,qp2,qp3,...]          (shells, nb, nb) array of nb x nb matricies
        #            q = [q1,q2,q3,...]             (shells)         array of floats
        #        rdens = [rdens1,rdens2,rdens3,...] (shells, nb)     array of nb vectors
        #     wnew_arr = [wnew1,wnew2,wnew3,...]    (shells)         array of floats
        #     enum_arr = [enum1,enum2,enum3,...]    (shells, nb, nb) array of arrays of floats
        ################################COMMENTS##########################################

        # I don't need this...
        # data = load('test1.npz')
        # data = load('g0inv_data/Na400beta20.npz')

        # Check the trend vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        trend = False
        if num > 2:
            print("Comparing radius of", previous_element, "with", two_elements_ago)
            differences[two_ele_ago_num] = radii[prev_ele_num]-radii[two_ele_ago_num]
            # If the trend difference is within 5% of Ashcroft's trend difference, then set consecutive_trend_errors = 0
            error = 0.05
            lower_bound = (1-error)*Ashcroft_trend[two_ele_ago_num]
            expected_value = Ashcroft_trend[two_ele_ago_num]
            calculated_value = differences[two_ele_ago_num]
            upper_bound = (1+error)*Ashcroft_trend[two_ele_ago_num]
            within_bounds = abs(lower_bound) <= abs(calculated_value) <= abs(upper_bound)
            same_sign = sign(calculated_value) == sign(expected_value)
            # Tracing Print Statements
            # print("Lower Bound:", lower_bound)
            # print("Calculated value:", calculated_value)
            # print("Upper Bound:", upper_bound)
            # print("Within bounds:", within_bounds)
            # print("Same sign:", same_sign)
            if (within_bounds and same_sign):
                trend = True
                consecutive_trend_errors = 0
            else:
                consecutive_trend_errors += 1
            if consecutive_trend_errors >= 2:
                # diff_threshold = 4 # Threshold to reach in the difference array to start saving g0inv values
                # if (differences[diff_threshold] > 0) or (differences[diff_threshold] < 0):
                #     THIS BLOCK OF CODE CURRENTLY DOESN'T WORK!!!
                #     # g0inv_arr_saves.append(g0inv)
                #     new_row = {'g0inv':g0inv, 'Radii':radii, 'Differences':differences}
                #     g0inv_data = g0inv_data.append(new_row, ignore_index=True)
                #     # g0inv_data.to_csv("g0inv values and data" + experiment + ".csv")
                print("Not enough trend fits")
                print("********************************Outputs********************************")
                print("|Current g0inv:", g0inv)
                for i in range(max_ele):
                    if radii[i] == 0:
                        break
                    else:
                        print("|Radius of", elements[i], ":", radii[i])
                print("|The difference array is", differences)
                print("***********************************************************************")
                # If you want to output/save something everytime the trend fails, this is where to do it!

                # Save data to the dataframes, then concat dataframes and save to file
                # This code is very memory inefficient...
                new_row_g0inv = {'g0inv':g0inv}
                g0inv_data = g0inv_data.append(new_row_g0inv, ignore_index=True)
                new_row_radii = {k:v for (k,v) in zip(elements,radii)}
                radii_data = radii_data.append(new_row_radii, ignore_index=True)
                new_row_differences = {k:v for (k,v) in zip(differences_data.columns, differences)}
                differences_data = differences_data.append(new_row_differences, ignore_index=True)
                g0inv_radii_differences_data = pd.concat([g0inv_data,radii_data,differences_data],axis=1)
                g0inv_radii_differences_data.to_csv("g0inv values and data" + experiment + ".csv")
                print("Data saved to " + "g0inv values and data" + experiment + ".csv")

                print(g0inv_radii_differences_data)
                consecutive_trend_errors = 0
                break
        # Check the trend ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        # Load intial data from file...
        if not(num == 1):
            print("Loading file "+previous_element+experiment+'.npz')
            data = load(previous_element+experiment+'.npz') # Load data using electron number
            wb = data['wb']
            shell_dens = data['shell_dens']  # Shells density components.
            dens = sum(shell_dens)  # Total density components.
            nb = size(wb,1)
            nb_arr = zeros(shells, dtype = int)
            for i in range(0,shells):
                nb_arr[i] = nb*(i+1)
        # Don't forget to turn off 'dens' before loop. Line 126

        # # Increase basis functions... I didn't modify this, not compatible!
        # nb2 = 200
        # wb1 = append(wb1,zeros(nb2))
        # wb2 = append(wb2,zeros(nb2))
        # wb3 = append(wb3,zeros(nb2))
        # dens1 = append(dens1,zeros(nb2))
        # dens2 = append(dens2,zeros(nb2))
        # dens3 = append(dens3,zeros(nb2))
        # dens = dens1+dens2+dens3
        # nb = nb+nb2
        # nb2 = 2*nb
        # nb3 = 3*nb

        # # ... or reduce number of basis functions... I didn't modify this, not compatible!
        # nb = 200
        # wb1 = data['wb1'][0:nb]
        # wb2 = data['wb2'][0:nb]
        # wb3 = data['wb3'][0:nb]
        # dens1 = data['dens1'][0:nb]  # Shell1 density components.
        # dens2 = data['dens2'][0:nb]  # Shell2 density components.
        # dens3 = data['dens3'][0:nb]  # Shell3 density components.
        # dens = dens1+dens2+dens3  # Total density components.
        # nb2 = 2*nb
        # nb3 = 3*nb

        # ... or set intial data by hand.
        if num == 1:
            print("Setting Initial Data")
            # nb = 400  # Number of basis functions. 
            nb_arr = zeros(shells, dtype = int)
            for i in range(0,shells):
                nb_arr[i] = nb*(i+1)
            wb = zeros((shells,nb)) # Initial field component guess zero.
        # Don't forget to turn on 'dens' before loop. Line 126

        # Real space setup...
        nrr = 10000                  # Number of real space radial grid points.
        rc = 10.0                    # Cutoff distance (size of spherical box).
        rr = linspace(0.0001,rc,nrr) # Define real space r-axis.
        realn_arr = zeros((shells,nrr)) # Initial shell densities set to zero.
        realn_arr[0] = exp(-2.0*rr)/pi  # Initial shell1 density set to hydrogen density.

        # Set numerical parameters for computation.
        itmax = 200             # Maximum number of self-consistent iterations.
        tol = 1.0e-9            # Convergence tolerance.
        mix = 0.05              # Picard iteration mixing parameter.
        nblist = arange(0,nb)   # Indices run from 0 to nb-1.
        nbtrue = arange(1,nb+1) # Indices running from 1 to nb.

        # Set physical parameters.
        # beta = 20.0   # Default # 1/kBT, temperature/contour parameter in energy units of Hartree.
        beta = beta_arr[num-1]              # 1/kBT, temperature/contour parameter in energy units of Hartree.
        print("Current beta:", beta)
        vol =  (4.0*pi*rc**3)/3.0 # Volume of spherical box.
        dens0_arr = zeros(shells) # Initialize dens0_arr array
        for i in range(0,shells):
            dens0_arr[i] = shell_config[i]/vol # Average number density of electrons in each shell.
        # dens0 = num/vol                        # Average total number density of electrons.
        # pauli_correction = (dens0)**(1/3) # Pauli Correction Term

        # g0inv = 5.742468000 # Default # Define inverse density of states for uniform electron gas.
        # g0inv = 5.742468000   # Define inverse density of states for uniform electron gas.
        # gamma = findgamma(nb) # Pack gamma from fortran.

        def basis(n):   # Basis functions.
            bas = sqrt(2.0/3.0)*(rc/rr)*sin(n*pi*rr/rc)
            return bas
            
        # Define external potential components.
        wext = array([-num*sqrt(6.0)/(rc*n*pi) for n in nbtrue]) # Need to subtract num/rc from free energy due to finite boundary.  
        # Turn off dens if loading data from file!!!
        if num == 1:
            shell_dens = zeros((shells,nb))
            shell_dens[0] = array([(3.0/(rc**3))*trapz(realn_arr[0]*(rr**2)*basis(n+1),rr) for n in nblist])  # Find density components from real space data.
            dens = sum(shell_dens)
            print("Setting dens variables")
        # Turn off dens if loading data from file!!!

        # Alternate comments between the two radius variables!
        # radius = 0
        # radius_old = -1
        # difference = radius - radius_old
        #radius = load(current_element+experiment+'_RADIUS.npz')['radius']

        # while abs(difference) > 10**(-4):
        # volume = 4/3 * pi * radius**3
        pauli_correction = 1 # Pauli Correction Term
        # if not(radius == 0):
        #     pauli_correction = (num/volume)**(1/3) # Pauli Correction Term
        # Add Pauli-Edwards-Flory-Huggins excluded volume potentials.
        wp = zeros((shells,nb))          # Initialize wp array
        for i in range(0,shells):
            wp[i] = dens - shell_dens[i] # Shell i+1 Pauli components.
        wp *= g0inv/pauli_correction     # Shell i+1 Pauli components.

        def anderson(dev): # Anderson matrix and vector definitions.
            umat = zeros((anders-1,anders-1))
            vvec = zeros((anders-1,1))
            for m1 in range(anders-1):
                for m2 in range(anders-1):
                    umat[m1,m2] = sum((dev[0,:]-dev[m1+1,:])*(dev[0,:]-dev[m2+1,:]))
                vvec[m1] = sum((dev[0,:]-dev[m1+1,:])*dev[0,:])
            avec = solve(umat,vvec)
            return avec

        anders = 2      # Guess + Anderson histories count.
        andtol = 1.0e-1
        #andtol = 2.5e-1     # When to try an Anderson step.

        whis = zeros((anders,nb_arr[-1])) # Initialize field output histories.
        whis[0,:] = consolidate(wb)
        dev = zeros((anders,nb_arr[-1]))  # Initialize deviation function histories.
        devtot = 10000.0*ones(anders)     # Initialize deviation total histories.

        it = 0
        while devtot[0]>tol and it<=itmax:
            it = it+1
            Amat = zeros((shells,nb,nb)) # Initialize Amat array
            val  = zeros((shells,nb))    # Initialize val  array
            vec  = zeros((shells,nb,nb)) # Initialize vec  array
            vect = zeros((shells,nb,nb)) # Initialize vect array
            for i in range(0,shells):
                # Build A matrix for shells. 
                Amat[i] = diag(-0.5*((nbtrue*pi/rc)**2)) - einsum('ijk,k->ij',gamma,whis[0,(i*nb):nb_arr[i]])                           
                val[i], vec[i] = eigh(Amat[i]) # Find eigenvalues and normalized eigenvectors
                vect[i] = transpose(vec[i])    # Find transpose of eigenvector matrix
            # Dval1 = diag(exp(beta*val1))  # Diagonal matrix of exponentiated eigenvalues for shell1.   
            # Dval2 = diag(exp(beta*val2))  # Diagonal matrix of exponentiated eigenvalues for shell2.
            # Dval3 = diag(exp(beta*val3))  # Diagonal matrix of exponentiated eigenvalues for shell3.
            # Use below values instead if larger beta needed than float64 can handle in exponents. 
            Dval  = zeros((shells,nb,nb)) # Initialize: Diagonal matrix of exponentiated eigenvalues for shells. 
            qp    = zeros((shells,nb,nb)) # Initialize: Find the propagator (partial partition function) components qp (qp = UDU^T) for shells.   
            q     = zeros(shells)         # Initialize: Single particle partition function for shells.
            rdens = zeros((shells,nb))    # Initialize: Raw density components shells. 
            for i in range(0,shells):
                Dval[i]       = diag(exp(float64(beta*val[i])))        # Diagonal matrix of exponentiated eigenvalues for shells. 
                qp[i]         = matmul(vec[i],matmul(Dval[i],vect[i])) # Find the propagator (partial partition function) components qp (qp = UDU^T) for shells.   
                q[i]          = sum(diag(qp[i]))                       # Single particle partition function for shells.
                rdens[i]      = einsum('ijk,ij->k',gamma,qp[i])        # Raw density components shells. 
                shell_dens[i] = dens0_arr[i]*rdens[i]/q[i]             # Density components shells.  
            dens = sum(shell_dens, axis=0) # Total density components.  
            
            # Classical electron-electron (Hartree) term.
            wee = array([4.0*dens[n-1]*(rc**2)/(pi*(n**2)) for n in nbtrue]) 
            wee = ((num-1)/num)*wee  # Alternative to exchange: just substract Fermi self-interaction.

            # Add Pauli-Edwards-Flory-Huggins excluded volume potentials.
            wp = zeros((shells, nb))         # Initialize wp
            for i in range(0,shells):
                wp[i] = dens - shell_dens[i] # Shell i+1 Pauli components.
            wp *= g0inv/pauli_correction     # Shell i+1 Pauli components.
            wnew_arr = wext+wee+wp           # Field components for each shell (array initialization)
            wnew = consolidate(wnew_arr)     # All field components together.              
            dev = roll(dev,1,axis=0)         # Store old deviation functions.
            devtot = roll(devtot,1,axis=0)   # Store old deviation totals.
            dev[0] = wnew-whis[0,:]          # New deviation functions.
            dev2 = sum(dev[0]**2)
            norm = sum(wnew**2)
            devtot[0] = dev2/norm            # New deviation total.
            perdevtot = abs(max(devtot[1:anders])-devtot[0])/max(devtot[1:anders])
            whis = roll(whis,1,axis=0)       # Store old output fields.
            whis[0,:] = wnew
            
            if (perdevtot<andtol and it>anders and devtot[0]<0.1):     # Decide whether to do an Anderson step. 
                avecmix = anderson(dev)
                for j in range(anders-1):
                    wnew = wnew+(avecmix[j]*(whis[j+1,:]-whis[0,:])) 
                print('Anderson step.')   
            else:
                whis[0,:] = mix*wnew+(1.0-mix)*whis[1,:] # Simple mixing.
            print(it, devtot[0])

        # Convert to real space to check result.
        temp    = [dens[n]*basis(n+1) for n in nblist]
        realn   = sum(temp,axis=0) # Real space electron density.
        # realn   = zeros(nrr)
        # for i in range(0,nrr):
        #     if i % 10000 == 0:
        #         print("Calculating realn (", i, "/", nrr, ")")
        #     for j in range(0,nb):
        #         realn[i] += temp[j][i]
        # peak       = max(realn)
        # radius_old = radius
        index      = 0
        for i in realn:
            if i < 0.001:
                radius = rr[index]
                break
            index += 1
        radii[num-1] = angstroms(radius)
        # difference = radius - radius_old
        # print("peak = ", peak)
        # print("index = ", index)
        # print("radius = ", radius)
        # print("difference = ", difference)

        fe = 0                               # Free Energy Initialization
        realn_arr = zeros((shells, nrr))
        for i in range(0,shells):
            print("Calculating realn_arr (", (i+1), "/", shells, ")")
            temp = [shell_dens[i,n]*basis(n+1) for n in nblist]
            realn_arr[i] = sum(temp, axis=0)
            # for j in range(0,nrr):
            #     for k in range(0,nb):
            #         realn_arr[i][j] += temp[k][j]  # Shells real space electron density.
            fe -= shell_config[i]*log(q[i])/beta + 0.5*vol*sum(shell_dens[i]*wp[i])/pauli_correction
        fe -= num/rc + 0.5*vol*sum(dens*wee) # Free Energy.

        temp = [whis[0,n]*basis(n+1) for n in nblist]
        realw = sum(temp,axis=0)    # Real space field.
        # realw = zeros(nrr)
        # for i in range(0,nrr):
        #     if i % 10000 == 0:
        #         print("Calculating realw (", i, "/", nrr, ")")
        #     for j in range(0,nb):
        #         realw[i] += temp[j][i]
        temp = [wnew[n]*basis(n+1) for n in nblist]
        realwnew = sum(temp,axis=0) # Real space field again.
        # realwnew = zeros(nrr)
        # for i in range(0,nrr):
        #     if i % 10000 == 0:
        #         print("Calculating realwnew (", i, "/", nrr, ")")
        #     for j in range(0,nb):
        #         realwnew[i] += temp[j][i]

        # realconv = trapz((realw-realwnew)**2,rr)/trapz(realwnew**2,rr)
        realconv = trapz((realw-realwnew)**2,rr)  # Real space unnormalized convergence of fields.

        # # Radius and volume of each pair calculations
        # pair1_radius = 0
        # # Pair 1 radius
        # for i in realn_arr[0]:
        #     if i < 0.001:
        #         #pair1_radius = rr[realn_arr[0].index(i)]
        #         pair1_radius = rr[where(realn_arr[0] == i)[0][0]]
        #         break
        # # Pair 1 volume
        # pair1_volume = 4 * pi * pair1_radius**3 / 3
        # # Pair 1 Density
        # pair1_density = shell_config[0]/pair1_volume


        # # Every other pair radius
        # inner_radii = zeros(shells - 1)
        # outer_radii = zeros(shells - 1)
        # for i in range(1,shells):
        #     index = 0
        #     for j in realn_arr[i]:
        #         if (index == 0) and (j > 0.001):
        #             #index = realn_arr[i].index(j)
        #             index = where(realn_arr[i] == j)[0][0]
        #             inner_radii[i-1] = rr[index]
        #         if not(index == 0) and (j < 0.001):
        #             #outer_radii[i-1] = rr[realn_arr[i].index(j)]
        #             outer_radii[i-1] = rr[where(realn_arr[i] == j)[0][0]]
        #             break
        # # Every other pair volume
        # pairN_volumes = 4 * pi * (outer_radii**3 - inner_radii**3) / 3
        # # Every other pair density
        # pairN_densities = zeros(shells - 1)
        # for i in range(1,shells):
        #     pairN_densities[i-1] = shell_config[i]/pairN_volumes[i-1]

        # Outputs.
        end = time.time()                             # Run time.
        enum = trapz(4.0*pi*(rr**2)*realn,rr)         # Total electron number. 
        enum_arr = trapz(4.0*pi*(rr**2)*realn_arr,rr) # Shells electron number. 

        print('Computation time:', end - start)
        print('Convergence:', devtot[0])
        print('Real space unnormalized convergence:', realconv)
        print('Electron number:', enum)
        for i in range(0,shells):
            print('Shell'+ str(i+1) + ' electron number:', enum_arr[i])

        # # Print the radii
        # print('Pair 1 Radius:', pair1_radius)
        # for i in range(1,shells):
        #     print('Pair ' + str(i+1) + ' Inner Radius:', inner_radii[i-1])
        #     print('Pair ' + str(i+1) + ' Outer Radius:', outer_radii[i-1])

        # # Print the volumes
        # print('Pair 1 Volume:', pair1_volume)
        # for i in range(1,shells):
        #     print('Pair ' + str(i+1) + ' Volume:', pairN_volumes[i-1])

        # # Print the Densities
        # print('Pair 1 Density:', pair1_density)
        # for i in range(1,shells):
        #     print('Pair ' + str(i+1) + ' Density:', pairN_densities[i-1])

        print('Radius:', radius)
        print('Free energy:', fe)
        # Save the radius and run the code again
        # savez(current_element+experiment+'_RADIUS.npz', radius = radius)

        # Save the electron densities for each shell for later graphing
        savez(current_element+experiment+'SHELLS.npz', realn = realn, realn_arr = realn_arr)

        # Plotting total electron density
        # fig = pl.figure()
        # pl.plot(rr, 4.0*pi*(rr**2) * realn, total_line, label='Total Electron Density')
        # pl.title("Total Radial Electron Density for " + current_element + experiment)
        # pl.xlabel("Radius from the nucleus, [r / a.u.]")
        # pl.ylabel("Electron Density, [4.0*pi*(rr**2) * n(r) / a.u.]")
        # pl.legend(loc='best')
        # pl.show()
        # fig.savefig('Total Electron Density for' + current_element + experiment + '.png')

        # Plotting pure electron densities shell by shell
        # fig = pl.figure()
        # pl.plot(rr, realn, total_line, label='Total Electron Density')
        # for i in range(0,shells):
        #     pl.plot(rr, realn_arr[i], shell_lines[i], label='Shell ' + str(i+1) + ' Density')
        # pl.title("Pure Electron Shell Density for " + current_element + experiment)
        # pl.xlabel("Radius from the nucleus, [r / a.u.]")
        # pl.ylabel("Electron Density, [n(r) / a.u.]")
        # pl.legend(loc='best')
        # pl.show()
        # fig.savefig('Pure Electron Shell Density for ' + current_element + experiment + '.png')

        # Plotting electron densities shell by shell (This was uncommented before)
        # fig = pl.figure()
        # pl.plot(rr, 4.0*pi*(rr**2) * realn, total_line, label='Total Electron Density')
        # for i in range(0,shells):
        #     pl.plot(rr, 4.0*pi*(rr**2) * realn_arr[i], shell_lines[i], label='Shell ' + str(i+1) + ' Density')
        # pl.title("Electron Shell Density for " + current_element + experiment)
        # pl.xlabel("Radius from the nucleus, [r / a.u.]")
        # pl.ylabel("Electron Density, [4.0*pi*(rr**2) * n(r) / a.u.]")
        # pl.legend(loc='best')
        # #pl.show()
        # fig.savefig('Electron Shell Density for ' + current_element + experiment + '.png')

        # Plotting electron densities shell by shell zoomed in
        # fig = pl.figure()
        # pl.plot(rr, 4.0*pi*(rr**2) * realn, total_line, label='Total Electron Density')
        # for i in range(0,shells):
        #     pl.plot(rr, 4.0*pi*(rr**2) * realn_arr[i], shell_lines[i], label='Shell ' + str(i+1) + ' Density')
        # pl.title("Electron Shell Density for " + current_element + " (zoomed in)" + experiment)
        # pl.xlabel("Radius from the nucleus, [r / a.u.]")
        # pl.ylabel("Electron Density, [4.0*pi*(rr**2) * n(r) / a.u.]")
        # pl.ylim(bottom=0,top= 1.6)
        # pl.legend(loc='best')
        # pl.show()
        # fig.savefig('Electron Shell Density for' + current_element + experiment + ' (zoomed in).png')

        # #savez('P800beta80.npz',dens1=dens1,dens2=dens2,dens3=dens3,wb1=whis[0,0:nb],wb2=whis[0,nb:nb2],wb3=whis[0,nb2:nb3]) # Save results to a file.

        # Save arrays with the name of the element. These arrays are used for calculating the next element
        for i in range(0,shells):
            wb[i] = whis[0,(i*nb):nb_arr[i]]
        savez(current_element+experiment+'.npz', shell_dens = shell_dens, wb = wb) # Save results to a file. These arrays will be used for calculating the next element
        print("Saved shell densities to file", current_element+experiment+'.npz')
    
    # Break conditions for the while(True) loop vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    if(num == 18): # Break when it reaches Argon
        break

    # if(g0inv <= 0):
    #     break

    if(g0inv >= 2000): # Break when g0inv reaches a critical value
        break
    print("Warning, while loop is not breaking!!!!!!") # A reminder that I created an endless while loop
    # Break conditions for the while(True) loop ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Save results to file
g0inv_radii_differences_data = pd.concat([g0inv_data,radii_data,differences_data],axis=1)
g0inv_radii_differences_data.to_csv("g0inv values and data" + experiment + ".csv")
print("Data saved to " + "g0inv values and data" + experiment + ".csv") 
print("The final g0inv is", g0inv) 
print("The differences array is", differences)