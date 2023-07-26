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
import time

start = time.time()

# The shell_config array represents the number of electrons in each shell. This is dynamic
# DISCLAIMER: Make sure the length of shell_config is the 
# same as the number of shells you will need for your final element
# For example, if you require 7 shells then set shell_config for Hydrogen as
# shell_config = [1,0,0,0,0,0,0]
# Goal for C   = [2,2,1,1]
shell_config   = [2,1,0,0]         # Configuration of electrons
shells         = len(shell_config) # Number of shells
num            = sum(shell_config) # Total number of electrons in the atom.

# Lines for the graphs
total_line = 'b-'
shell_lines = ['k--','g-','r--','c:','m-','y--','b:'] # For plots: Need to add enough colours in this array to match the shell config! These lines are for graphing!


# These variables are for naming files and plots:
elements = ["Hydrogen", "Helium", "Lithium", "Beryllium", "Boron", "Carbon", "Nitrogen", "Oxygen", "Fluorine", "Neon", 
"Sodium", "Magnesium", "Aluminium", "Silicon","Phosphorus","Sulfur","Chlorine","Argon"]
current_element = elements[num-1]    # Calculate the current element using the electron number
previous_element = elements[num-2]   # Calculate the previous element using the electron number
experiment = "_approximations" # Name your experiment, files will be saved with this string suffixed to it

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

# Load intial data from file...
if not(num == 1):
    print("Loading file")
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
    nb = 400  # Number of basis functions. 
    nb_arr = zeros(shells, dtype = int)
    for i in range(0,shells):
        nb_arr[i] = nb*(i+1)
    wb = zeros((shells,nb)) # Initial field component guess zero.
    print("Setting Initial Data")
# Don't forget to turn on 'dens' before loop. Line 126

# Real space setup...
nrr = 10000                 # Number of real space radial grid points.
rc = 10.0                    # Cutoff distance (size of spherical box).
rr = linspace(0.0001,rc,nrr) # Define real space r-axis.
realn_arr = zeros((shells,nrr))  # Initial shell densities set to zero.
realn_arr[0] = exp(-2.0*rr)/pi   # Initial shell1 density set to hydrogen density.

# Set numerical parameters for computation.
itmax = 200             # Maximum number of self-consistent iterations.
tol = 1.0e-9            # Convergence tolerance.
mix = 0.05              # Picard iteration mixing parameter.
nblist = arange(0,nb)   # Indices run from 0 to nb-1.
nbtrue = arange(1,nb+1) # Indices running from 1 to nb.

# Set physical parameters.
# beta = 20.0   # Default # 1/kBT, temperature/contour parameter in energy units of Hartree.
beta = 20.0               # 1/kBT, temperature/contour parameter in energy units of Hartree.
vol =  (4.0*pi*rc**3)/3.0 # Volume of spherical box.
dens0_arr = zeros(shells) # Initialize dens0_arr array
for i in range(0,shells):
    dens0_arr[i] = shell_config[i]/vol # Average number density of electrons in each shell.
# dens0 = num/vol                        # Average total number density of electrons.
# pauli_correction = (dens0)**(1/3) # Pauli Correction Term

# g0inv = 5.742468000 # Default # Define inverse density of states for uniform electron gas.
g0inv = 5.742468000   # Define inverse density of states for uniform electron gas.
gamma = findgamma(nb) # Pack gamma from fortran.

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

def consolidate(arr_of_arr): # Takes a list of arrays and consolidates row-wise (along the first axis) it into a 1D array 
        rows = size(arr_of_arr,0)
        cons = []
        for i in range(0,rows):
            cons = np.append(cons,arr_of_arr[i])
        return cons # Example: [[1,2,3],[4,5,6],[7,8,9]] -> [1,2,3,4,5,6,7,8,9]

# Alternate comments between the two radius variables!
radius = 0
radius_old = -1
difference = radius - radius_old
#radius = load(current_element+experiment+'_RADIUS.npz')['radius']

while abs(difference) > 10**(-4):
    volume = 4/3 * pi * radius**3
    pauli_correction = 1 # Pauli Correction Term
    if not(radius == 0):
        pauli_correction = (num/volume)**(1/3) # Pauli Correction Term
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

    # Free arrays I no longer need
    # import gc
    # del Amat
    # del Dval
    # del vec
    # del vect
    # del qp
    # # gc.collect()

    # Convert to real space to check result.
    temp    = [dens[n]*basis(n+1) for n in nblist]
    realn   = sum(temp,axis=0) # Real space electron density.
    # realn   = zeros(nrr)
    # for i in range(0,nrr):
    #     if i % 10000 == 0:
    #         print("Calculating realn (", i, "/", nrr, ")")
    #     for j in range(0,nb):
    #         realn[i] += temp[j][i]
    peak       = max(realn)
    radius_old = radius
    index      = 0
    for i in realn:
        if i < 0.001:
            radius = rr[index]
            break
        index += 1
    difference = radius - radius_old
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

# Outputs.
end = time.time()                             # Run time.
enum = trapz(4.0*pi*(rr**2)*realn,rr)         # Total electron number. 
enum_arr = trapz(4.0*pi*(rr**2)*realn_arr,rr) # Shells electron number. 

print('Computation time:', end - start)
print('Convergence:', devtot[0])
print('Real space unnormalized convergence: ', realconv)
print('Electron number:', enum)
for i in range(0,shells):
    print('Shell'+ str(i+1) + ' electron number:', enum_arr[i])
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

# Plotting electron densities shell by shell
fig = pl.figure()
pl.plot(rr, 4.0*pi*(rr**2) * realn, total_line, label='Total Electron Density')
for i in range(0,shells):
    pl.plot(rr, 4.0*pi*(rr**2) * realn_arr[i], shell_lines[i], label='Shell ' + str(i+1) + ' Density')
pl.title("Electron Shell Density for " + current_element + experiment)
pl.xlabel("Radius from the nucleus, [r / a.u.]")
pl.ylabel("Electron Density, [4.0*pi*(rr**2) * n(r) / a.u.]")
pl.legend(loc='best')
pl.show()
fig.savefig('Electron Shell Density for ' + current_element + experiment + '.png')

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

# Save file with the name of the element
for i in range(0,shells):
    wb[i] = whis[0,(i*nb):nb_arr[i]]
savez(current_element+experiment+'.npz', shell_dens = shell_dens, wb = wb) # Save results to a file. These arrays will be used for calculating the next element
#print(current_element+experiment+'.npz')