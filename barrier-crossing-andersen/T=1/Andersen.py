from __future__ import print_function, division
import numpy as np
#from mdtraj.utils import in_units_of
from copy import deepcopy
import time
import math as ma
from random import *
import random as random

M_PI = 3.14

Temp = 1
Position = 0.0
Velocity = 0.0
ConservedEnergy = 1.0

U = 0.0
OldF = 0.0
F = 0.0
NumberOfSteps = 1000
Tstep = 0.1
NumberOfInitializationSteps = 100
Nu = 10
Freqnos = 4
NumberOfNoseHooverChains = 5
Cstart = 1.0
M=0
nc=1
count = 2

Choice = 2	#Choose the system type# #1.NVE,2.Andersen,3.NoseHoover,4.MonteCarlo,5.Berendsen#

MAXIMUM_NOSEHOOVER_CHAINLENGTH = 100

############################################
#	Force Calculation
############################################
#calculate the forces and potential energy
def Force(x,Ufs,Ffs):
	global Uf,Ff
	if(x<=0.0):
		Uf=2.0*ma.pow(M_PI,2.0)*ma.pow(x,2.0)
		Ff=-4.0*ma.pow(M_PI,2.0)*x
	elif(x>=1.0):
		Uf=2.0*ma.pow(M_PI,2.0)*ma.pow(x-1.0,2.0)
		Ff=-4.0*ma.pow(M_PI,2.0)*(x-1.0)
	else:
		Uf=1.0-ma.cos(2.0*M_PI*x)
		Ff=-2.0*M_PI*ma.sin(2.0*M_PI*x)
	return Uf,Ff

############################################
#	Min function
############################################
def MIN(a,b):
	if float(a)>=float(b):
		return float(b)
	else:
		return float(a)

############################################
#	Switch
############################################
class switch(object):
	def __init__(self, value):
		self.value = value
		self.fall = False

	def __iter__(self):
		yield self.match
		raise StopIteration

	def match(self, *args):
		if self.fall or not args:
			return True
		elif self.value in args:
			self.fall = True
			return True
		else:
			return False

############################################
#	Random Velocity
############################################
#Generates A Random Velocity According To A Boltzmann Distribution

def RandomVelocity(temp):
	sigma=np.sqrt(temp)
	v=np.random.normal(0, sigma)
	return v


############################################
#	NVE ensemble
############################################

def IntegrateNVE():
	global U,F,OldF,Position,Velocity,Tstep,ConservedEnergy
	NewVelocity = 0.0
	NewPosition = 0.0
	## start
	#start modification
	ForceBack = Force(Position,U,F)
	OldF = ForceBack[1]
	NewPosition = Position + Velocity*Tstep + OldF*0.5*ma.pow(Tstep, 2.0) #Velocity-Verlet integrator#
	ForceBack = Force(NewPosition,U,F)
	NewVelocity = Velocity + 0.5*(ForceBack[1]+OldF)*Tstep 

	Velocity = NewVelocity
	Position = NewPosition
	## end modification
	ConservedEnergy = ForceBack[0] + 0.5*ma.pow(Velocity,2.0) 

############################################
#	Calculate diffusion
############################################

Tmax = 1000
T0max = 200
dis = 0
oldp = 0
time = 0
D = []
Tt0 = np.zeros(T0max+1, dtype=float) 
R2t = np.zeros(Tmax+1, dtype=float)
Rx0 = np.zeros(T0max+1, dtype=float) 
Nvacf = np.zeros(Tmax+1, dtype=float)

def SampleDiff(Sitch):
	global dis, oldp, time, D
	Tcounter = 0
	i = 0
	T = 0
	Dt = 0
	Tvacf = 0	
	T0 = 0
	Dtime = 0.0

	for case in switch(Sitch):
		if case(1):
			Tvacf=0
			T0=0
			for i in range(1,Tmax+1):
				R2t[i]=0.0
				Nvacf[i]=0.0
			break
		if case(2):
			if D == []:
				dis += Position	
			else:
				dis += (Position - oldp)
			oldp = Position
			time += 1
			if (time%1000==0) or time==1:
				D.append((dis**2)/(2*time*Tstep))
			break

		if case(3):
			FilePtr=open('diffussion.dat','w+')
			for i in range(len(D)):
				if i >=1 :
					print('D: '+str(D[i]), file=FilePtr)
			aveD = np.mean(D[1:])
			print('Ave D: '+str(aveD), file=FilePtr)

			FilePtr.close()
			break

############################################
#	Sample Prof
############################################
Maxx = 500
Dens = np.zeros(Maxx+1,dtype=float)
Dtot = 0.0
Delta = 0.0
def SampleProf(Sitch):
	global Dtot, Delta, Position, Dens

	for case in switch(Sitch):
		if case(1):
			Delta=(Maxx-1.0)/4.0
			Dtot=0.0
			for i in range(0,Maxx):
				Dens[i]=0.0
			break
		if case(2):
			Dtot+=1.0
			i=int(Delta*(Position+2.0))
			if ( (i>=0) and (i<Maxx)):
				Dens[i]+=1.0
			break
		if case(3):
			FilePtr=open('density.dat','w+')
			for i in range(0,Maxx):
				if Dens[i]>0:
					print('%f %f' % (i/Delta-2.0,Dens[i]/Dtot), file=FilePtr)
			FilePtr.close()
			break

###########################################
#	Initialize Nose-Hoover Chain
###########################################

w = np.zeros(5,dtype=float)
QMass = np.zeros(5,dtype=float)
xlogs = np.zeros(5,dtype=float)
vlogs = np.zeros(5,dtype=float)
glogs = np.zeros(5,dtype=float)
UNoseHoover = 0.0

def InitializeNoseHoover():
	global U, F, Tstep, Position, ConservedEnergy, Velocity, OldF, M, nc, UNoseHoover, QMass, xlogs, vlogs, glogs, Freqnos, w
	
	M=NumberOfNoseHooverChains
	nc=1

	w[0]=1.0/(4.0-pow(4.0,1.0/3.0))
	w[1]=w[0]
	w[2]=1.0-4.0*w[0]
	w[3]=w[0]
	w[4]=w[0]

	QMass[0]=Temp/ma.pow(Freqnos,2.0)
	QMass[1]=Temp/ma.pow(Freqnos,2.0)
	QMass[2]=Temp/ma.pow(Freqnos,2.0)
	QMass[3]=Temp/ma.pow(Freqnos,2.0)
	QMass[4]=Temp/ma.pow(Freqnos,2.0)

	xlogs[0]=0.0
	xlogs[1]=0.0
	xlogs[2]=0.0
	xlogs[3]=0.0
	xlogs[4]=0.0

	vlogs[0]=1.0/QMass[0]
	vlogs[1]=1.0/QMass[1]
	vlogs[2]=1.0/QMass[2]
	vlogs[3]=1.0/QMass[3]
	vlogs[4]=1.0/QMass[4]

	glogs[0]=0.0
	glogs[1]=(QMass[0]*ma.pow(vlogs[0],2.0)-Temp)/QMass[1]
	glogs[2]=(QMass[1]*ma.pow(vlogs[1],2.0)-Temp)/QMass[2]
	glogs[3]=(QMass[2]*ma.pow(vlogs[2],2.0)-Temp)/QMass[3]
	glogs[4]=(QMass[3]*ma.pow(vlogs[3],2.0)-Temp)/QMass[4]

## NoseHoover thermostat (NVT-Ensemble) by Glenn J. Martyna
## Ref: Molecular Physics 1996, Vol. 87, No. 5, 1117-1157

def NoseHooverNVT(dt):
	global U, F, Tstep, Position, ConservedEnergy, Velocity, OldF, M, nc, UNoseHoover, w, QMass, xlogs, vlogs, glogs
	i = 0
	j = 0
	k = 0
	AA = 0.0
	scale=0.0
	UKinetic_pressure = 0.0

	scale = 1.0
	UKinetic_pressure = ma.pow(Velocity,2.0)
	glogs[0] = (UKinetic_pressure-Temp)/QMass[0]
	for i in range(0,nc):
		for j in range(1,6):
			vlogs[M-1] += (glogs[M-1]*w[j-1]*dt)/(4.0*nc)

			for k in range(0,M-1):
				AA = np.exp(-(w[j-1]*dt/(8.0*nc))*vlogs[M-k-1])
				vlogs[M-k-2] = vlogs[M-k-2]*ma.pow(AA,2.0)+glogs[M-k-2]*AA*w[j-1]*dt/(4.0*nc)

			AA = np.exp(-(w[j-1]*dt/(2.0*nc))*vlogs[0])
			scale = scale*AA
			glogs[0] =(ma.pow(scale,2.0)*UKinetic_pressure-Temp)/QMass[0]

			for k in range(1,M+1):
				xlogs[k-1] += vlogs[k-1]*w[j-1]*dt/(2.0*nc)
			for k in range(0,M-1):
				AA = np.exp(-(w[j-1]*dt/(8.0*nc))*vlogs[k+1])
				vlogs[k] = vlogs[k]*ma.pow(AA,2.0)+glogs[k]*AA*(w[j-1]*dt/(4.0*nc))
				glogs[k+1] = (QMass[k]*ma.pow(vlogs[k],2.0)-Temp)/QMass[k+1]
			vlogs[M-1] += glogs[M-1]*w[j-1]*dt/(4.0*nc)

	## scale velocity
	Velocity*=scale

	## compute thermostat energy
	UNoseHoover=0.0
	for i in range(0,M):
		UNoseHoover += 0.5*ma.pow(vlogs[i],2.0)*QMass[i]+Temp*xlogs[i]

## integrate the equations of motion using an explicit Nose-Hoover chain
## see the work of Martyna/Tuckerman et al.

def IntegrateNoseHoover():
	global U, F, Tstep, Position, ConservedEnergy, Velocity, OldF, M, nc, UNoseHoover
	U=0.0
	F=0.0

	## thermostat 0.5xTstep
	NoseHooverNVT(Tstep)

	## normal integration
	Velocity += 0.5*Tstep*OldF
	Position += Tstep*Velocity
	ForceBack = Force(Position,U,F)
	U = ForceBack[0]
	F = ForceBack[1]
	OldF = F
	Velocity += 0.5*Tstep*OldF

	## thermostat 0.5xTstep
	NoseHooverNVT(Tstep)

	ConservedEnergy = U+0.5*ma.pow(Velocity,2.0)+UNoseHoover

###########################################
#	IntegrateBerendsen
###########################################

def IntegrateBerendsen():
	global U, F, Temp, Tstep, Velocity, Position, OldF, ConservedEnergy
	Ekin = 0.0
	tau = 0.1
	Berendsen = 0.0
	Stoch = 0.0
	Scale = 0.0
	lambdas = 0.0
	PastPosition = 0.0
	FutuVelocity = 0.0
	PastPosition = 0.0
	FutuPosition = 0.0

	# Integration of equations of motion: Leap Frog.
	ForceBack = Force(Position,U,F)
	U = ForceBack[0]
	F = ForceBack[1]
	
	#begin modification
	#determine lambdas
	Ekin = ma.pow(Velocity,2.0)
	lambdas = ma.sqrt(1+(Tstep/tau)*(Temp/Ekin-1))
	#end modification
	NewVelocity = lambdas*(Velocity+Tstep*F)
	Position += Tstep*NewVelocity
	# Conserved quantity is v(t) = (Vpos + Vnew)/2.
	# We square them, so we need a factor 8.
	ConservedEnergy = U + ma.pow(NewVelocity+Velocity,2.0)/8.0	#0.5 * 
	#sqrt(0.5*(NewVelocity + Velocity), 2.0) #
	Velocity = NewVelocity
	OldF = F

##################################################################
#	integrate the equations of motion for the Andersen thermostat
##################################################################

def IntegrateAndersen():
	global U, F, Temp, Tstep, Velocity, Position, OldF, ConservedEnergy, Nu
	Momentum = 0.0
	FutuPosition = 0.0
	NewVelocity = 0.0
	sigma = 0.0
	
	# integrate the equations of motion for the Andersen thermostat
	# start modification
	ForceBack = Force(Position, U, OldF)
	OldF = ForceBack[1]
	FutuPosition = Position + Velocity*Tstep + 0.5*OldF*ma.pow(Tstep,2.0)
	NewVelocity = Velocity + OldF*0.5*Tstep
	Position = FutuPosition
	Velocity = NewVelocity
	#Velocity Verlet#
	if random.random() <= (Nu*Tstep):
		Velocity = RandomVelocity(Temp)
	# end modification
	#ConservedEnergy=U+ma.pow(Velocity, 2.0)/2.0

###########################################
#	## MC simulation
###########################################
def IntegrateMonteCarlo():
	global Uold, F, Temp, Tstep, Velocity, Position, OldF, ConservedEnergy
	Uold = 0.0
	Unew = 0.0
	F = 0.0
	NewPosition = 0.0

	ForceBack =	Force(Position,Uold,F)
	Uold = ForceBack[0]
	F = ForceBack[1]
	NewPosition = Position+2.5*(random.random()-0.5)
	ForceBack = Force(NewPosition,Unew,F)
	Unew = ForceBack[0]
	F = ForceBack[1]

	if (random.random()<ma.exp(-(Unew-Uold)/Temp)):
		Position = NewPosition
		Uold = Unew
	OldF = 0.0
	Velocity = 0.0
	ConservedEnergy = 0.0

###########################################
#	ReadData
###########################################

def ReadData():

	global U,OldF,Position,Temp,Velocity,Nu,NumberOfNoseHooverChains, Choice
	global Freqnos,NumberOfSteps,NumberOfInitializationSteps,Tstep

	FilePtr = open('input','w+')
	Velocity = RandomVelocity(Temp)

	ForceBack = Force(Position,U,OldF)
	U = ForceBack[0]
	OldF = ForceBack[1]

	print('Md Of A Particle Over A Energy Barrier', file=FilePtr)
	print('Number of Steps (x 1000): %d' % NumberOfSteps, file=FilePtr)
	print('Number of Init Steps  : %d' % NumberOfInitializationSteps, file=FilePtr)
	print('Time step             : %f' % Tstep, file=FilePtr)
	print('Temp           : %f' % Temp, file=FilePtr)

	for case in switch(Choice):
		if case(2):
			print('Collision Frequency   : %f' % Nu,file=FilePtr)
			break
		if case(3):
			Velocity = ma.sqrt(Temp)
			InitializeNoseHoover()
			print('Number of Nose-Hoover Chains   : %d' % NumberOfNoseHooverChains, file=FilePtr)
			print('Frequency Omega               : %f' % Freqnos, file=FilePtr)
			break
		else:
			break
	FilePtr.close()

############################################
#	Mdloop
############################################


def MdLoop():
	global Choice, Cstart, ConservedEnergy, Position, Velocity, count
	FilePtr=open('pos_vel.dat','w+')
	
	Cc1 = 1.0
	Cc2 = 1.0
	Tt1 = 1.0
	Tt2 = 1.0
	countT = 1
	SampleDiff(1)
	SampleProf(1)

	CycleMultiplication = 1000
	PrintFrequency = 200

	for i in range (0,NumberOfSteps):
		for j in range(0,CycleMultiplication):
			print('Progress is %f %%' % (((i*CycleMultiplication+j)*100.0)/(1.0*NumberOfSteps*CycleMultiplication)))
			for case in switch(Choice):
				if case(1):
					IntegrateNVE()
					break
				if case (2):
					IntegrateAndersen()
					break
				if case (3):
					IntegrateNoseHoover()
					break
				if case (4):
					IntegrateMonteCarlo()
					break
				if case (5):
					IntegrateBerendsen()
					break
			if (i>NumberOfInitializationSteps):
				if Choice==4:
					SampleDiff(2)
					SampleProf(2)
					Tt1 += ma.pow(Velocity,2.0)
					Tt2 += 1.0
					if (j%PrintFrequency)==0:
						print('%d %f %f' % (count,Position,Uold), file=FilePtr)
						count += 1
				else:
					SampleDiff(2)
					SampleProf(2)
					Tt1 += ma.pow(Velocity,2.0)
					Tt2 += 1.0
					if (j%PrintFrequency)==0:
						print('%d %f %f' % (count,Position,Velocity), file=FilePtr)
						count += 1
			if (i==NumberOfInitializationSteps):
				Cstart = ConservedEnergy
			elif(i>NumberOfInitializationSteps):
				if((Choice!=2) and (Choice!=4)):
					Cc1 += ma.fabs((ConservedEnergy-Cstart)/Cstart)
					Cc2 += 1.0

	SampleDiff(3)
	SampleProf(3)
	print('Energy Drift          : %f' % (Cc1/Cc2), file=FilePtr)
	print('Average Temp.         : %f' % (Tt1/Tt2), file=FilePtr)
	FilePtr.close()

############################################
#	Barrier
############################################
def barrier():
	ReadData()	# read input parameters
	MdLoop()		#MD run
if __name__==  '__main__':
	barrier()
