# orbital-free-density-funcitonal-theory

The code in this repository extends code that was used to explore atomic shell structure from an orbital free density functional theory from this paper: <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.012813>. This is code that I worked on while I conducted research with Russell Thompson from the University of Waterloo in the department of Physics.

## Installation

1. Install all required Python packages by running this command in terminal:

```{bash}
pip install -r requirements.txt
```

2. Fortran90 is required. Recommended method of installation: 
   1. Go to <http://www.cygwin.org/cygwin/>, download and run `setup-x86_64.exe`. 
   2. When you get to the package selector, change the View to Full. 
   3. Search `gcc-fortran` and select it for installation.
3. The code requires the library `findgamma`, the included Fortran code, to be compiled. To compile it, run this command in the terminal:

```{bash}
python -m numpy.f2py -c gammafile.f90 -m findgamma --opt='-Ofast'; cp ./findgamma/.libs/* .
```
