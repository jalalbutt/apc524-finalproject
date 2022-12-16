"""
APC 524 
Final Project
Quantifying resilience of coupled networks

@author Jalal Butt
Date: 12-03-22


Module file handling the perturbation network
----------------------------------------------


Outputs:
--------
	- time-evolution of perturbation
		--> entirely independent of what's happening on the grid. I.e., need not output 
			individual time-steps
"""



## (Elliptic Equations hw)
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
# from scipy import signal

from dataclasses import dataclass
import traceback


## Arb parameters for testing.

## external inputs, to be fed in from 
"""
m = 100 # number of grid-points
n = 100
n = 100 #  currently not implemented
L_x = 10
H_y = 10

f_type = 'oscillatory'
f_type = 'delta'
source_point = (-2,-2)
source_strength = 1000
radius = 10
"""


@dataclass
class PerturbedNetwork:
	"""
	Class for the perturbation network model. 

	Inputs:
	------
		f_type :   forcing function type (from a finite set)
		m      :   # of grid points
		radius :   perturbation extent in terms of grid [points] 
		source_center :   coordinate pair; % of (L_x, H_y) (not indices)


	Outputs:
	-------
		U :   np.array (2D if len(timesteps)== 1); 3D if > 1)

	Methods:
	--------
		construct_A:   construct the spatial operator
		construct_source:   construct perturb source (several options)
		solve_laplacian_system:   numerically solve laplacian
		compute_discretization_error:   <only valid for an analytic RHS>
		plot_solution:   plot the solution
		plot_error:    plot the error

		static_solve:   compute the time-independent solution


	Usage:
	-----
	Example:
		m = 100 # number of grid-points
		n = 100
		n = 100 #  currently not implemented
		L_x = 10
		H_y = 10

		f_type = 'oscillatory'
		f_type = 'delta'
		source_center=  = ()
		source_strength = 1000
		radius = m/L_x* 0.1

		pertNet2 = PerturbedNetwork(f_type= f_type, source_point= source_center, 
		                            m= m, L_x= L_x, H_y= H_y, radius= radius)
		U = pertNet2.static_solve()
		pertNet2.plot_solution()


	"""

	## Inputs:
	f_type: str
	source_center: [tuple,list,np.ndarray]  
	m: int
	L_x: int
	H_y: int
	radius: float = m/L_x* 0.01
	source_strength: float = 1000

	def __post_init__(self):  # what should i do w this? generate tests?
		self.failure = False
		self.delta_x = L_x / (m + 1)
		self.x       = np.linspace(0, self.L_x, self.m + 2)
		self.y       = np.linspace(0, self.H_y, self.m + 2)

		try:
			assert (len(source_point) == 2), "Source point must be coordinate pair"
			assert (m > 10), "Grid must be AT LEAST 10x10"
			assert (radius < 0.1*m), "Disturbance radius cannot span >10% of grid"
			assert (f_type in ['delta', 'oscillatory']), "Requested source-type is not yet configured."
			assert (radius > 0.6*delta_x), "Source cannot be resolved on the grid due to small radius"
			
		except:
			traceback.print_exc()
			self.failure= True
			self.intialization_failure_message = "Cannot complete this action due to initialization failure"





	# Construct A
	def construct_A(self, m, delta_x):
		"""
		Construct the 5-point Laplacian operator.

		Inputs:
		------
		m [int] 
		delta_x [float] : spatial-spacing

		Outputs:
		-------
		A [sparse obj] : 2D 5-point Laplacian operator

		Improvements TBD:
		----------------
		[] Square grid --> rectangular grid
		"""
		e = np.ones(m)
		T = sparse.spdiags([e, -4.0 * e, e], [-1, 0, 1], m, m)
		S = sparse.spdiags([e, e], [-1, 1], m, m)
		I = sparse.eye(m)
		A = sparse.kron(I, T) + sparse.kron(S, I)
		A /= delta_x**2

		return A



	# Right-hand-side (forcing-function)
	def construct_source(self, x,y, f_type= 'oscillatory',
	                     source_strength= 1, impulse_extent= 30, source_center= (10,12), 
	                     radius= 2):
		"""
		Construct source (i.e. bomb source)
		
		Inputs:
		------
		x, y [np.array]  : x and y arrays, to be put into mesh form
		f_type [str]     : forcing-function type (oscillatory, localized, etc.)
			- oscillatory could correspond to a heat-wave.
		source_point     : 
		source_strength  : 
		radius : 

		Outputs:
		-------
		X, Y [np.array, [len(x) x len(y)]] : x,y mesh
		f [np.array, [len(x) x len(y)] ]   : discrete forcing function
		
		"""
		X_source, Y_source = np.meshgrid(x[1:-1], y[1:-1])

		if f_type.lower() == 'oscillatory':
			f = -2.0 * np.sin(X_source) * np.sin(Y_source)
		elif f_type.lower() == 'delta':
			# f = np.zeros(X_source.shape)
			# sp = source_point
			# for i in range(impulse_extent):
			# 	f += signal.unit_impulse(X_source.shape, sp)*source_strength
			# 	sp = (sp[0]-1, sp[1]-1)
			circle = np.sqrt((X_source-source_center[1]*self.L_x)**2 \
			                 + (Y_source-source_center[0]*self.H_y)**2)
			f      = np.zeros(X_source.shape)
			f[circle<radius] += source_strength

		else:
			print(f_type + " doesn't exist yet as a source constructor.")

		return X_source,Y_source,f


	



	# Solve discretized PDE in time
	def solve_laplacian_system(self, m, A, f):
		"""
		Solve the Poisson eq. w/ the respective forcing-function.

		Inputs:
		-------
		m : # points 
		A : discretized laplacian
		f : forcing function

		Outputs:
		-------
		U [np.array]  : heat-kernel
		"""
		U = np.zeros((m+2, m+2))
		U[1:-1, 1:-1] = linalg.spsolve(A, f.reshape(m**2, order='F')).reshape((m, m), order='F')

		return U




	def compute_discretization_error(self, x,y,U):
		"""
		Compute numerical error introduced by discretization.

		**Currently only works for the oscillatory solution**

		Inputs:
		-------
		x,y [np.array, grid]  : 
		U [np.array, grid]    : 

		Outputs:
		-------
		X,Y
		norm_error
		grid_error

		"""
		# Error
		X, Y = np.meshgrid(x, y)
		norm_error = np.linalg.norm((x[1] - x[0]) * (U - np.sin(X) * np.sin(Y)), ord=1)


		grid_error = np.abs(U - np.sin(X) * np.sin(Y))

		print("Normalization error" + str(norm_error))

		return X,Y, norm_error, grid_error




	def plot_solution(self):#, X,Y,U, L_x, H_y):
		if self.failure: return print(self.intialization_failure_message)

		# Plot solution
		fig1 = plt.figure()
		axes1 = fig1.add_subplot(1, 1, 1)
		sol_plot = axes1.pcolor(self.X, self.Y, self.U, cmap=plt.get_cmap('RdBu_r'))
		axes1.set_title("Solution u(x,y)")
		axes1.set_xlabel("x")
		axes1.set_ylabel("y")
		axes1.set_xlim((0.0, self.L_x))
		axes1.set_ylim((0.0, self.H_y))
		cbar1 = fig1.colorbar(sol_plot, ax=axes1)
		cbar1.set_label("u(x, y)")

		plt.show()


	def plot_error(self):#, X,Y,norm_error,grid_error):
		fig2 = plt.figure()
		axes2 = fig2.add_subplot(1, 1, 1)
		sol_plot = axes2.pcolor(self.X, self.Y, self.grid_error, 
		                        cmap=plt.get_cmap('RdBu_r'))
		axes2.set_title("Error |U - u|")
		axes2.set_xlabel("x")
		axes2.set_ylabel("y")
		axes2.set_xlim((0.0, self.L_x))
		axes2.set_ylim((0.0, self.H_y))
		cbar2 = fig2.colorbar(sol_plot, ax=axes2)
		cbar2.set_label("u(x, y)")

		plt.show()
	



	# Laplacian solution
	def static_solve(self):
		"""
		Solution to the laplacian
		
		Inputs:
		-------
		self 

		Outputs:
		-------
		U

		"""
		if self.failure: return print(self.intialization_failure_message)

		self.A = self.construct_A(self.m, self.delta_x)

		# what is this X,Y used for besides source construction?
		self.X_source,self.Y_source,\
		self.f = self.construct_source(self.x,self.y,f_type = self.f_type, 
		                                      source_strength= self.source_strength,
		                                      source_center= self.source_center, 
		                                      radius= self.radius)
		self.U = self.solve_laplacian_system(self.m,self.A,self.f)

		self.X,self.Y,self.norm_error,\
		self.grid_error = self.compute_discretization_error(self.x,self.y,self.U)

		return self.U

