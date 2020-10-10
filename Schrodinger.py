'''

Given a spherically symmetric potential, solve the radial Schrodinger equation. 
Obtain numerically the eigenenergy and its corresponding eigenfunction.

with runge-kutta method
'''
import numpy as np
import matplotlib.pyplot as plt 
from sys import exit

# define constants
hbar = 1.05*1e-34          # plank constant in [J*s]
me = 9.109*1e-31           # electron rest mass [kg]
C = hbar**2/(2*me)         
joule_to_eV = 6.242*1e18
eV_to_joule = 1.602*1e-19
permittivity = 8.854*1e-12 # vacuum permittivity
q = 1.6*1e-19              # [Coloumb]

l=0                        # only consider the case where the quantum number l = 0 

# define the box 
angstr = 1e-10             # [angstroms]
x = 0.0
xStop = 10*angstr 
L = xStop - x              # box size

# runge kutta module
def integrate(F,x,y,xStop,h):
    def run_kut4(F,x,y,h):
        K0 = h*F(x,y)
        K1 = h*F(x + h/2.0, y + K0/2.0)
        K2 = h*F(x + h/2.0, y + K1/2.0)
        K3 = h*F(x + h, y + K2)
        return (K0 + 2.0*K1 + 2.0*K2 + K3)/6.0
    X = []
    Y = [] 
    X.append(x) 
    Y.append(y) 
    while x < xStop:
        h = min(h,xStop - x)
        y = y + run_kut4(F,x,y,h) 
        x=x+h
        X.append(x) 
        Y.append(y)
    return np.array(X),np.array(Y)

# get nth eigenenergy and plot eigenfunction
def calculate(E_lower_limit):
    node = 0             # the number of nodes initialized as zero
    nodeArray = []       # the number of nodes for each energy values will be put into the array 
    energyBoundary = []  # every time the number of nodes changes, the corresponding energy value will be put into the array

    E = 0.0        # energy value initialized as zero
    Estep = 0.01   # energy value initialized as zero # energy step size in iteration
    nLoop = 0      # the number of iteration

    energyBoundary.append(E_lower_limit) # the first eigenenergy is bigger than E_lower_limit

    # find the range where nth eigenvalue lies 
    while (node < n):
        E = E_lower_limit + Estep*nLoop # from the lower limit energy value, iterate, increasing the E value
        def energy_in_joule(): # energy value in [J] 
            return energy_in_eV()*eV_to_joule
        def energy_in_eV(): # energy value in [eV] 
            return E

        # runge-kutta 
        def F(x,y):
            # radial Schrodinger equation
            F = np.zeros(2)
            F[0] = y[1]
            F[1] = ((potentialVeff(x)-energy_in_joule())/C)*y[0] 
            return F

        # at r = 0; y[1]: some arbitrary value
        y = np.array([0.0, 1.0*angstr])

        h = 0.1*angstr                 # integration step 
        X,Y = integrate(F,x,y,xStop,h) # integrate

        # the number of nodes 
        node = 0
        # find the number of roots 
        for i in range(len(Y)-1):
            if ( np.sign(Y[i,0]) != np.sign(Y[i+1,0]) ): 
                node = node + 1
        node = node -1 # exclude the root r = 0
        nodeArray.append(node) # for every E value, mark the number of nodes
        # if the number of nodes changes, mark the corresponding energy value, putting it into an array
        if ( len(nodeArray) > 1 ):
            if ( nodeArray[nLoop-1] != nodeArray[nLoop] ): 
                energyBoundary.append(round(E,4))

        nLoop = nLoop + 1 # the number of iteration

    # find the eigenvalue in the range
    box_boundary = [] 
    energy = energyBoundary[n-1] # from this value, (whose corresponding number of node == the number of node for En) we start iteration

    nstep = 0 # the number of iteration 
    while (energy < energyBoundary[n]):
        energy = (energyBoundary[n-1] + Estep*nstep) # in the range, iterate, increasing the energy value
        energy_in_joule = energy*eV_to_joule 
        # iterate over energy values in the range 
        if (energy != energyBoundary[n]):
            # runge-kutta 
            def G(x,y):
                G = np.zeros(2)
                G[0] = y[1]
                G[1] = ((potentialVeff(x)-energy_in_joule)/C)*y[0] 
                return G
            X1,Y1 = integrate(G,x,y,xStop,h)

            box_boundary.append(abs( Y1[len(Y1)-1,0] - 0.0)) # for each energy values, calculate the distance from x axis at the right boundary, and put it into an array
        nstep = nstep + 1 # the number of iteration

    index = np.argmin(box_boundary) # find the index in the array for the energy value whose corresponding wave function satisfies the boundary condition the most; its distance from x axis at the right boundary is the smallest
    eigenvalue = energyBoundary[n-1] + index*Estep # using the index, obtain the eigenenergy value

    # when E < 0 & n is large enough for the absolute value of eigenenergy to approximate to zero; difference between adjacent eigenvalues are much smaller than the integration step size, terminate the program with alert
    if ( np.sign(energyBoundary[n]) != np.sign(energyBoundary[0]) and np.sign(energyBoundary[0]) != 0 ):
        print("\nPlease enter a n value smaller than",n) 
    # otherwise
    else:
        print("\nEnergy eigenvalue at n =",n," ->",round(eigenvalue,2),"eV") 
        print("Box size: width =",L)

        # before normalization 
        R = []
        for i in range(len(Y)):
            R.append(Y[i,0]/(X[i]+1e-30))

        # normalization 
        Rnorm = []
        integ = 0.0
        for i in range(len(R)):
            Rnorm.append(R[i]**2*X[i]**2) 
            integ = integ + h*(Rnorm[i])

        Rafter = []
        for i in range(len(R)):
            Rafter.append(R[i]/integ)

        for i in range(len(R)): 
            Rnorm[i] = Rnorm[i]/integ

        # plot
        fig = plt.figure()
        title = "Radial Function ( n = " + str(n) + ", l = " + str(l) + " )" 
        plt.title(title,y=1.06)

        # normalized radial function
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(X,Rafter,'o')
        ax1.axhline(0,color="black")
        ax1.set_xlabel("r")
        ax1.set_ylabel("R(r)")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([]) 
        ax1.set_title("R(r)")
 
        # normalized radial distribution 
        ax2 = fig.add_subplot(1,2,2) 
        ax2.plot(X,Rnorm,'o-') 
        ax2.axhline(0,color="black") 
        ax2.set_xlabel("r") 
        ax2.set_ylabel("R(r)**2*r**2") 
        ax2.set_xticklabels([]) 
        ax2.set_yticklabels([]) 
        ax2.set_title("R(r)**2*r**2") 
        plt.show()


# Get User Input
if __name__ == "__main__":
    print("\nCentral Potential (r dependent potential), consider only the case when l = 0")
    print("1. Infinite Potential Well \n2. Hydrogen Atom \n")

    choice = int(input("Choose between the three options above - enter the number of your choice: "))
    
    if (choice == 1 or choice == 2):
        # eigenenergy index (nth eigenenergy)
        n = int(input("Enter n: "))
        # width of the box
        xStop = float(input("Enter box width L (usually, 1e-9 recommended): ")) 
        # box size corresponding to user input xStop 
        L = xStop - x
        # define potential
        def potentialVeff(r): 
            if (choice == 1):   # infinite well case
                vp = 0.0
            elif (choice == 2): # hydrogen atom case
                vp = -q**2/(4*np.pi*permittivity*r+1e-50) 
            return (vp+(C*l*(l+1)/(r**2+1e-50))) 
    else:
        print("\nPlease enter a valid input. Please run the program again. ") 
        exit()

    # First, try - energy lower limit value = 0; if this succeeds, E > 0; otherwise, it fails. In that case, retry with -10eV; if the retry fails, re-retry with -20eV; loop until you get the correct answer
    i=0
    while True: 
        try:
            calculate(-10*i) 
        except:
            i += 1
            continue 
        else:
            break
