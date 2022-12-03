"""
APC 524 
Final Project
Quantifying resilience of coupled Networks

@author Jalal Butt
Date: 12-03-22


Module file handling the perturbation network
----------------------------------------------


Inputs:
-------
	- grid obj 


Outputs:
--------
	- time-evolution of perturbation
		--> entirely independent of what's happening on the grid. I.e., need not output 
			individual time-steps
"""



## (Elliptic Equations hw)
import numpy
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg


## external inputs, to be fed in from 
m = 100 # grid-size
# n = 100 #  currently not implemented


# Problem specification
x = numpy.linspace(0, 2.0 * numpy.pi, m + 2)
y = numpy.linspace(0, 2.0 * numpy.pi, m + 2)
delta_x = 2.0 * numpy.pi / (m + 1)

# Construct A
e = numpy.ones(m)
T = sparse.spdiags([e, -4.0 * e, e], [-1, 0, 1], m, m)
S = sparse.spdiags([e, e], [-1, 1], m, m)
I = sparse.eye(m)
A = sparse.kron(I, T) + sparse.kron(S, I)
A /= delta_x**2

# Right-hand-side
X, Y = numpy.meshgrid(x[1:-1], y[1:-1])
f = -2.0 * numpy.sin(X) * numpy.sin(Y)

# Solve
U = numpy.zeros((m+2, m+2))
U[1:-1, 1:-1] = linalg.spsolve(A, f.reshape(m**2, order='F')).reshape((m, m), order='F')

# Error
X, Y = numpy.meshgrid(x, y)
print(numpy.linalg.norm((x[1] - x[0]) * (U - numpy.sin(X) * numpy.sin(Y)), ord=1))

# Plot solution
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
sol_plot = axes.pcolor(X, Y, U, cmap=plt.get_cmap('RdBu_r'))
axes.set_title("Solution u(x,y)")
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_xlim((0.0, 2.0 * numpy.pi))
axes.set_ylim((0.0, 2.0 * numpy.pi))
cbar = fig.colorbar(sol_plot, ax=axes)
cbar.set_label("u(x, y)")

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
sol_plot = axes.pcolor(X, Y, numpy.abs(U - numpy.sin(X) * numpy.sin(Y)), cmap=plt.get_cmap('RdBu_r'))
axes.set_title("Error |U - u|")
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_xlim((0.0, 2.0 * numpy.pi))
axes.set_ylim((0.0, 2.0 * numpy.pi))
cbar = fig.colorbar(sol_plot, ax=axes)
cbar.set_label("u(x, y)")
plt.show()
