#!/usr/bin/env python

from random import *
import numpy as np
import math as ma
import time
import random

#Main Part of MD simulation#

print('Start simulation\n')

#Set the initial Parameters#
NumParticles=1000	#Number of particles#
Boxlength=10.0		#Box length unit: angstrom#
Temp=0.5			#Temperature unit: Kelvin#
Rcut=2.5			#Cut off randius=0.5*Boxlength#
Sigma=1.0			#Reduced value for sigma#
Epilson=1.0			#Reduced value for epilson#
Ecut=4.0*(pow(Rcut,-12.0)-pow(Rcut,-6.0))				#Energy cut off#
NumMDStep=50000		#Simulation steps for MD run#
NumMDInit=5000		#Simulation steps for initialization#
Tstep=0.005			#Time step, unit: femtosecond#
M_PI=3.1415			#Value of pi#
mu=50
Ttarget=10

Coord_XOld=np.zeros(NumParticles,dtype=np.float)		#Old X-direction coordinate of particle#
Coord_X=np.zeros(NumParticles,dtype=np.float)			#X-direction coordinate of particle#
Coord_XNew=np.zeros(NumParticles,dtype=np.float)		#New X-direction coordinate of particle#
Coord_XNonPDB=np.zeros(NumParticles,dtype=np.float)		#X-direction coordinate of particle without PDB treatment#

Coord_YOld=np.zeros(NumParticles,dtype=np.float)		#Old Y-direction coordinate of particle#
Coord_Y=np.zeros(NumParticles,dtype=np.float)			#Y-direction coordinate of particle#
Coord_YNew=np.zeros(NumParticles,dtype=np.float)		#New Y-direction coordinate of particle#
Coord_YNonPDB=np.zeros(NumParticles,dtype=np.float)		#Y-direction coordinate of particle without PDB treatment#


Coord_ZOld=np.zeros(NumParticles,dtype=np.float)		#Old Z-direction coordinate of particle#
Coord_Z=np.zeros(NumParticles,dtype=np.float)			#Z-direction coordinate of particle#
Coord_ZNew=np.zeros(NumParticles,dtype=np.float)		#New Z-direction coordinate of particle#
Coord_ZNonPDB=np.zeros(NumParticles,dtype=np.float)		#Z-direction coordinate of particle without PDB treatment#

Velocities_X=np.zeros(NumParticles,dtype=np.float)		#Velocities of particles along X-direction#
Velocities_Y=np.zeros(NumParticles,dtype=np.float)		#Velocities of particles along Y-direction#
Velocities_Z=np.zeros(NumParticles,dtype=np.float)		#Velocities of particles along Z-direction#

Force_X=np.zeros(NumParticles,dtype=np.float)			#Force of particles along X-direction#
Force_Y=np.zeros(NumParticles,dtype=np.float)			#Force of particles along Y-direction#
Force_Z=np.zeros(NumParticles,dtype=np.float)			#Force of particles along Z-direction#
ForceOld_X=np.zeros(NumParticles,dtype=np.float)		#OldForce of particles along X-direction#
ForceOld_Y=np.zeros(NumParticles,dtype=np.float)		#OldForce of particles along Y-direction#
ForceOld_Z=np.zeros(NumParticles,dtype=np.float)		#OldForce of particles along Z-direction#

MomentumX=0.0
MomentumY=0.0
MomentumZ=0.0

UKinetic=0.0
UPotential=0.0
Pressure=0.0

MaxNum=int(pow(NumParticles,1.0/3.0)+1.5)
LatticeGap=Boxlength/(MaxNum+2.0)						#Generate lattice distribution of particles#

Place=0.2*LatticeGap									#Used in steepest decent algorithm# 

IndPart=0
#Generate Initial Coordinates of system#
for IndX in range(0,MaxNum):
	for IndY in range(0,MaxNum):
		for IndZ in range(0,MaxNum):
			if IndPart<NumParticles:
				Coord_X[IndPart]=(IndX+0.01*(random.random()-0.5))*LatticeGap
				Coord_Y[IndPart]=(IndY+0.01*(random.random()-0.5))*LatticeGap
				Coord_Z[IndPart]=(IndZ+0.01*(random.random()-0.5))*LatticeGap
			IndPart+=1

#Generate Initial Velocities of system#
for IndPart in range(0,NumParticles):
	Velocities_X[IndPart]=random.random()
	Velocities_Y[IndPart]=random.random()
	Velocities_Z[IndPart]=random.random()
	MomentumX+=Velocities_X[IndPart]
	MomentumY+=Velocities_Y[IndPart]
	MomentumZ+=Velocities_Z[IndPart]

MomentumX/=NumParticles
MomentumY/=NumParticles
MomentumZ/=NumParticles

#Calculate kinetic energy#
for IndPart in range(0,NumParticles):
	Velocities_X[IndPart]-=MomentumX
	Velocities_Y[IndPart]-=MomentumY
	Velocities_Z[IndPart]-=MomentumZ

	UKinetic+=(pow(Velocities_X[IndPart],2.0)+pow(Velocities_Y[IndPart],2.0)+pow(Velocities_Z[IndPart],2.0))

#Scale all velocities to the correct temperature#
Scale=ma.sqrt(((3.0*NumParticles-3.0)*Temp)/UKinetic)
for IndPart in range(0,NumParticles):
	Velocities_X[IndPart]*=Scale
	Velocities_Y[IndPart]*=Scale
	Velocities_Z[IndPart]*=Scale

#Calculate better positions using a steepest decent algorithm#
for IndOpt in range(0,50):
	if IndOpt==0:
		#Initilization#
		for IndPart in range(0,NumParticles):
			Force_X[IndPart]=0.0
			Force_Y[IndPart]=0.0
			Force_Z[IndPart]=0.0

		#Calculate potential energy and pressure#
		for IndPartA in range(0,NumParticles-1):
			for IndPartB in range(IndPartA+1,NumParticles):
				DelX=Coord_X[IndPartA]-Coord_X[IndPartB]
				DelY=Coord_Y[IndPartA]-Coord_Y[IndPartB]
				DelZ=Coord_Z[IndPartA]-Coord_Z[IndPartB]

				#Periodic boundary condition#
				DelX-=Boxlength*int(DelX/Boxlength)
				DelY-=Boxlength*int(DelY/Boxlength)
				DelZ-=Boxlength*int(DelZ/Boxlength)

				DisPair=ma.sqrt(pow(DelX,2)+pow(DelY,2)+pow(DelZ,2))
				#Check if the distance is within the cutoff radius#
				if DisPair<Rcut:
					UPotential+=4.0*(pow(DisPair,-12.0)-pow(DisPair,-6.0))-Ecut
					Forcef=48.0*(pow(DisPair,-6.0)-0.5*pow(DisPair,-3.0))
					Pressure+=Forcef
					Forcef=Forcef*pow(DisPair,-6.0)				#Calculate force on each particle#

					Force_X[IndPartA]+=Forcef*DelX
					Force_Y[IndPartA]+=Forcef*DelY
					Force_Z[IndPartA]+=Forcef*DelZ

					Force_X[IndPartB]-=Forcef*DelX
					Force_Y[IndPartB]-=Forcef*DelY
					Force_Z[IndPartB]-=Forcef*DelZ

		Pressure/=(3.0*pow(Boxlength,3))
		UPold=UPotential
	#Calculate maximum downhill gradient#
	MaxGra=0.0
	for IndPart in range(0,NumParticles):
		Coord_XOld[IndPart]=Coord_X[IndPart]
		Coord_YOld[IndPart]=Coord_Y[IndPart]
		Coord_ZOld[IndPart]=Coord_Z[IndPart]

		ForceOld_X[IndPart]=Force_X[IndPart]
		ForceOld_Y[IndPart]=Force_Y[IndPart]
		ForceOld_Z[IndPart]=Force_Z[IndPart]

		MomentumX=abs(Force_X[IndPart])
		MomentumY=abs(Force_Y[IndPart])
		MomentumZ=abs(Force_Z[IndPart])

		if MomentumX>MaxGra:
			MaxGra=MomentumX
		if MomentumY>MaxGra:
			MaxGra=MomentumY
		if MomentumZ>MaxGra:
			MaxGra=MomentumZ

	MaxGra=Place/MaxGra

	#Calculate improved coordinates#
	for IndPart in range(0,NumParticles):
		Coord_X[IndPart]+=MaxGra*Force_X[IndPart]
		Coord_Y[IndPart]+=MaxGra*Force_Y[IndPart]
		Coord_Z[IndPart]+=MaxGra*Force_Z[IndPart]

		if Coord_X[IndPart]>=Boxlength:
			Coord_X[IndPart]-=Boxlength
		if Coord_X[IndPart]<0.0:
			Coord_X[IndPart]+=Boxlength

		if Coord_Y[IndPart]>=Boxlength:
			Coord_Y[IndPart]-=Boxlength
		if Coord_Y[IndPart]<0.0:
			Coord_Y[IndPart]+=Boxlength

		if Coord_Z[IndPart]>=Boxlength:
			Coord_Z[IndPart]-=Boxlength
		if Coord_Z[IndPart]<0.0:
			Coord_Z[IndPart]+=Boxlength

	#Calculate New potential#
	for IndPart in range(0,NumParticles):
			Force_X[IndPart]=0.0
			Force_Y[IndPart]=0.0
			Force_Z[IndPart]=0.0

	#Calculate potential energy and pressure#
	for IndPartA in range(0,NumParticles-1):
		for IndPartB in range(IndPartA+1,NumParticles):
			DelX=Coord_X[IndPartA]-Coord_X[IndPartB]
			DelY=Coord_Y[IndPartA]-Coord_Y[IndPartB]
			DelZ=Coord_Z[IndPartA]-Coord_Z[IndPartB]

			#Periodic boundary condition#
			DelX-=Boxlength*int(DelX/Boxlength)
			DelY-=Boxlength*int(DelY/Boxlength)
			DelZ-=Boxlength*int(DelZ/Boxlength)

			DisPair=ma.sqrt(pow(DelX,2)+pow(DelY,2)+pow(DelZ,2))
			#Check if the distance is within the cutoff radius#
			if DisPair<Rcut:
				UPotential+=4.0*(pow(DisPair,-12.0)-pow(DisPair,-6.0))-Ecut
				Forcef=48.0*(pow(DisPair,-12.0)-0.5*pow(DisPair,-6.0))
				Pressure+=Forcef
				Forcef=Forcef*pow(DisPair,-6.0)				#Calculate force on each particle#

				Force_X[IndPartA]+=Forcef*DelX
				Force_Y[IndPartA]+=Forcef*DelY
				Force_Z[IndPartA]+=Forcef*DelZ

				Force_X[IndPartB]-=Forcef*DelX
				Force_Y[IndPartB]-=Forcef*DelY
				Force_Z[IndPartB]-=Forcef*DelZ

	Pressure/=(3.0*pow(Boxlength,3))

	if UPotential<UPold:
		UPold=UPotential
		Place*=1.2

		if Place>(0.5*Boxlength):
			Place=0.5*Boxlength
	if UPotential>=UPold:
		for IndPart in range(0,NumParticles):
			Force_X[IndPart]=ForceOld_X[IndPart]
			Force_Y[IndPart]=ForceOld_Y[IndPart]
			Force_Z[IndPart]=ForceOld_Z[IndPart]
			Coord_X[IndPart]=Coord_XOld[IndPart]
			Coord_Y[IndPart]=Coord_YOld[IndPart]
			Coord_Z[IndPart]=Coord_ZOld[IndPart]
		Place*=0.1
#Minimization done#

#Calculate previous coordinate using generated velocity#
for IndPart in range(0,NumParticles):
	Coord_XOld[IndPart]=Coord_X[IndPart]-Tstep*Velocities_X[IndPart]
	Coord_YOld[IndPart]=Coord_Y[IndPart]-Tstep*Velocities_Y[IndPart]
	Coord_ZOld[IndPart]=Coord_Z[IndPart]-Tstep*Velocities_Z[IndPart]

	Coord_XNonPDB[IndPart]=Coord_X[IndPart]
	Coord_YNonPDB[IndPart]=Coord_Y[IndPart]
	Coord_ZNonPDB[IndPart]=Coord_Z[IndPart]

#################################################################################
#           /\									#
#		   //\\								#
#		  //II\\							#
#           II 									#
#	Initialization done 							#
#################################################################################

Tempz=0.0				#Record real temperature#
dUtot=0.0				#Change of total energy#
dUtotAll=0.0				#Summation of energy drift#
Utot0=0.0				#Initial total energy#
UTotal=0.0				#Total energy#
ParaAv=np.zeros(6,dtype=np.float)	#Record important parameters:
						#Temp/Pressure/UKinetic/UPotential
						#UTotal/Times

for IndPara in range(0,6):
	ParaAv[IndPara]=0.0

#########################################
#Initialize radial distribution function#
#########################################
MaxNumRDF=500									#Maximum number of bins for RDF#
RDFDis=np.zeros(MaxNumRDF,dtype=np.float)
IndRDFStat=0.0									#Times to account RDF distribution#
for IndRDF in range(0,MaxNumRDF):
	RDFDis[IndRDF]=0.0
DeltaRDFGap=Boxlength/(2.0*MaxNumRDF)

##################################
#Initialize diffusion coefficient#
##################################
MaxNumTS=7500									#Maximum timesteps for the correlation time#
MaxNumT0=250									#Maximum number of time origins#
FreSeT0=50									#Frequency with which a new time origin is selected#
time=0										#Current time in units of Tstep# 
t0Counter=0									#Number of time origins stored so far#
t0time=np.zeros(MaxNumT0+1,dtype=np.int)					#Time of the stored time origin in units of Tstep#
t0Ind=0										#The index of the current t0 in the array of stored time origins#
CorrelTime=0									#Time difference between the current time and the time origin#
CumIntegration=0.0								#Cumlation of xx#
SampleCounter=np.zeros(MaxNumTS+1,dtype=np.int)					#Number of samples taken at a given CorrelTime#
Vacf=np.zeros(MaxNumTS+1,dtype=np.double)
R2=np.zeros(MaxNumTS+1,dtype=np.double)
Vxt0=np.zeros((NumParticles+1,MaxNumT0+1),dtype=np.double)
Vyt0=np.zeros((NumParticles+1,MaxNumT0+1),dtype=np.double)
Vzt0=np.zeros((NumParticles+1,MaxNumT0+1),dtype=np.double)
Rx0=np.zeros((NumParticles+1,MaxNumT0+1),dtype=np.double)
Ry0=np.zeros((NumParticles+1,MaxNumT0+1),dtype=np.double)
Rz0=np.zeros((NumParticles+1,MaxNumT0+1),dtype=np.double)

for IndDS in range(0,MaxNumTS):
	R2[IndDS]=0.0
	SampleCounter[IndDS]=0
	Vacf[IndDS]=0.0

EnergyDriftName='EnergyDrift.dat'
fEnD=open(EnergyDriftName,'w+')
EnDCount=1

MDSimParaName='MDSimulation.log'
fMDSim=open(MDSimParaName,'w+')

MDSimTraj='MDTraj.pdb'
fMDTraj=open(MDSimTraj,'w+')
TrajConfCount=0

atomName='P'	#atomname Particle#
resName='Par'	#resName Particle#
chainName='A'	#chainName A#

MDSimRDF='MDSimRDF.dat'
fMDRDF=open(MDSimRDF,'w+')

MDSimMSD='MDSimMSD.dat'
fMDSimMSD=open(MDSimMSD,'w+')

MDSimVACF='MDSimVACF.dat'
fMDSimVACF=open(MDSimVACF,'w+')

#Write initial structure#
for NParticlesSave in range(0,NumParticles):
	if NParticlesSave==0:
		print("REMARK IndOfConf=%d" % TrajConfCount, file=fMDTraj)	#Record index of conformation#
	line = "ATOM  %5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f %5.2f %5.2f      %-6s  " % ( NParticlesSave, atomName, resName, chainName, 
	NParticlesSave, Coord_X[NParticlesSave], Coord_Y[NParticlesSave], Coord_Z[NParticlesSave], 1.00, 0.00, atomName)
	print( line, file=fMDTraj)	#Coordinates of particles in 1st conf#
	if NParticlesSave==(NumParticles-1):
		print("END", file=fMDTraj)	#Record index of conformation#

#Start MD simulation& loop over all cycles#
for IndStep in range(0,NumMDStep):

	if IndStep%10==0:
		#Visualize the progress of simulation#
		print('\r Current progress:%.3f%%' % (IndStep*100/NumMDStep))

	###1. Calculate force#
	#Initilization#
	for IndPart in range(0,NumParticles):
		Force_X[IndPart]=0.0
		Force_Y[IndPart]=0.0
		Force_Z[IndPart]=0.0

	UPotential=0.0
	Pressure=0.0

	#Calculate potential energy and pressure#
	for IndPartA in range(0,NumParticles-1):
		for IndPartB in range(IndPartA+1,NumParticles):
			DelX=Coord_X[IndPartA]-Coord_X[IndPartB]
			DelY=Coord_Y[IndPartA]-Coord_Y[IndPartB]
			DelZ=Coord_Z[IndPartA]-Coord_Z[IndPartB]

			#Periodic boundary condition#
			DelX-=Boxlength*int(DelX/Boxlength)
			DelY-=Boxlength*int(DelY/Boxlength)
			DelZ-=Boxlength*int(DelZ/Boxlength)

			DisPair=np.double(ma.sqrt(pow(DelX,2)+pow(DelY,2)+pow(DelZ,2)))

			#Check if the distance is within the cutoff radius#
			if DisPair<Rcut:
				UPotential+=4.0*(pow(DisPair,-12.0)-pow(DisPair,-6.0))-Ecut
				Forcef=48.0*(pow(DisPair,-12.0)-0.5*pow(DisPair,-6.0))
				Pressure+=Forcef
				Forcef=Forcef*pow(DisPair,-6.0)				#Calculate force on each particle#

				Force_X[IndPartA]+=Forcef*DelX
				Force_Y[IndPartA]+=Forcef*DelY
				Force_Z[IndPartA]+=Forcef*DelZ

				Force_X[IndPartB]-=Forcef*DelX
				Force_Y[IndPartB]-=Forcef*DelY
				Force_Z[IndPartB]-=Forcef*DelZ

	Pressure/=(3.0*pow(Boxlength,3))	

	###2. Integrate the equations of motion#
	UKinetic=0.0
	for IndPart in range(0,NumParticles):
		Coord_XNew[IndPart]=2.0*Coord_X[IndPart]-Coord_XOld[IndPart]+Force_X[IndPart]*pow(Tstep,2)
		Coord_YNew[IndPart]=2.0*Coord_Y[IndPart]-Coord_YOld[IndPart]+Force_Y[IndPart]*pow(Tstep,2)
		Coord_ZNew[IndPart]=2.0*Coord_Z[IndPart]-Coord_ZOld[IndPart]+Force_Z[IndPart]*pow(Tstep,2)

		if random.random()<mu*Tstep:

                        Velocities_X[IndPart]=np.random.normal(loc=0.0, scale=Ttarget**(1/2),size=None)
                        Velocities_Y[IndPart]=np.random.normal(loc=0.0, scale=Ttarget**(1/2),size=None)
                        Velocities_Z[IndPart]=np.random.normal(loc=0.0, scale=Ttarget**(1/2),size=None)

		UKinetic+=0.5*(pow(Velocities_X[IndPart],2)+pow(Velocities_Y[IndPart],2)+pow(Velocities_Z[IndPart],2))
	#Note: For NumberOfSteps < NumberOfInitializationSteps; use the velocity scaling to get the exact temperature#
	#Otherwise: Scale=1.0 be aware that the positions/velocities have to be recalculated !!!!#

	#Scale=ma.sqrt(Temp*(3.0*NumParticles-3.0)/(2.0*UKinetic))

	UKinetic=0.0
	MomentumX=0.0
	MomentumY=0.0
	MomentumZ=0.0

	#Scale velocities and put particles back in the box#
	#Beware: the old positions are also put back in the box#

	for IndPart in range(0,NumParticles):
                
                MomentumX+=Velocities_X[IndPart]
                MomentumY+=Velocities_Y[IndPart]
                MomentumZ+=Velocities_Z[IndPart]

                Coord_XNew[IndPart]=Coord_XOld[IndPart]+2.0*Velocities_X[IndPart]*Tstep
                Coord_YNew[IndPart]=Coord_YOld[IndPart]+2.0*Velocities_Y[IndPart]*Tstep
                Coord_ZNew[IndPart]=Coord_ZOld[IndPart]+2.0*Velocities_Z[IndPart]*Tstep

                Coord_XNonPDB[IndPart]+=Coord_XNew[IndPart]-Coord_X[IndPart]
                Coord_YNonPDB[IndPart]+=Coord_YNew[IndPart]-Coord_Y[IndPart]
                Coord_ZNonPDB[IndPart]+=Coord_ZNew[IndPart]-Coord_Z[IndPart]

                UKinetic+=0.5*(pow(Velocities_X[IndPart],2)+pow(Velocities_Y[IndPart],2)+pow(Velocities_Z[IndPart],2))
                Coord_XOld[IndPart]=Coord_X[IndPart]
                Coord_YOld[IndPart]=Coord_Y[IndPart]
                Coord_ZOld[IndPart]=Coord_Z[IndPart]

                Coord_X[IndPart]=Coord_XNew[IndPart]
                Coord_Y[IndPart]=Coord_YNew[IndPart]
                Coord_Z[IndPart]=Coord_ZNew[IndPart]

		#Put particles back in the box#
		#The previous position has the same transformation (Why ?)#
                if Coord_X[IndPart]>=Boxlength:
                        Coord_X[IndPart]-=Boxlength
                        Coord_XOld[IndPart]-=Boxlength
                if Coord_X[IndPart]<0.0:
                        Coord_X[IndPart]+=Boxlength
                        Coord_XOld[IndPart]+=Boxlength

                if Coord_Y[IndPart]>=Boxlength:
                        Coord_Y[IndPart]-=Boxlength
                        Coord_YOld[IndPart]-=Boxlength
                if Coord_Y[IndPart]<0.0:
                        Coord_Y[IndPart]+=Boxlength
                        Coord_YOld[IndPart]+=Boxlength

                if Coord_Z[IndPart]>=Boxlength:
                        Coord_Z[IndPart]-=Boxlength
                        Coord_ZOld[IndPart]-=Boxlength

                if Coord_Z[IndPart]<0.0:
                        Coord_Z[IndPart]+=Boxlength
                        Coord_ZOld[IndPart]+=Boxlength
	Pressure+=2.0*UKinetic*NumParticles/(pow(Boxlength,3)*(3.0*NumParticles-3.0))
	#Integration done#
	
	if IndStep==1:
		print('Momentum X-Dir: %lf' % MomentumX,file=fMDSim)
		print('Momentum Y-Dir: %lf' % MomentumY,file=fMDSim)
		print('Momentum Z-Dir: %lf' % MomentumZ,file=fMDSim)
		print('Step %d Utotal: %lf UKinetic: %lf UPotential: %lf Temperature: %lf Pressure: %lf' % (IndStep-1,
			UTotal,UKinetic,UPotential,Temp,Pressure),file=fMDSim)

	UTotal=UKinetic+UPotential

	if IndStep%FreSeT0==0:
		if EnDCount==1:
			Utot0=UTotal
			print('Step %d EnergyDrift %f' % (EnDCount,abs((Utot0-UTotal)/Utot0)) , file=fEnD)
			dUtot=0.0
		if EnDCount>1:
			dUtot+=abs((Utot0-UTotal)/Utot0)
			print('Step %d EnergyDrift %f' % (EnDCount,dUtot/(EnDCount-1)) , file=fEnD)
		EnDCount+=1

	Tempz=2.0*UKinetic/(3.0*NumParticles-3.0)

	if ((IndStep>NumMDInit) and (IndStep%FreSeT0==0) ):
		print('Step %d Utotal: %lf UKinetic: %lf UPotential: %lf Temperature: %lf Pressure: %lf' % (IndStep,
			UTotal,UKinetic,UPotential,Tempz,Pressure),file=fMDSim)

	if IndStep%10==0:
		TrajConfCount+=1
		for NParticlesSave in range(0,NumParticles):
			if NParticlesSave==0:
				print("REMARK IndOfConf=%d" % TrajConfCount, file=fMDTraj)	#Record index of conformation#
			line = "ATOM  %5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f %5.2f %5.2f      %-6s  " % ( NParticlesSave, atomName, resName, chainName, 
							NParticlesSave, Coord_X[NParticlesSave], Coord_Y[NParticlesSave], Coord_Z[NParticlesSave], 1.00, 0.00, atomName)
			print( line, file=fMDTraj)	#Coordinates of particles in 1st conf#
			if NParticlesSave==(NumParticles-1):
				print("END", file=fMDTraj)	#Record index of conformation#

	if IndStep==(NumMDStep-1):
		print('Momentum X-Dir: %lf' % MomentumX,file=fMDSim)
		print('Momentum Y-Dir: %lf' % MomentumY,file=fMDSim)
		print('Momentum Z-Dir: %lf' % MomentumZ,file=fMDSim)

	#Calculate averages#
	if IndStep>NumMDInit:
		ParaAv[0]+=Tempz
		ParaAv[1]+=Pressure
		ParaAv[2]+=UKinetic
		ParaAv[3]+=UPotential
		ParaAv[4]+=UTotal
		ParaAv[5]+=1.0

		#Update total average Energy drift#
		dUtotAll+=(Utot0-UTotal)

	#Sample radial distribution function#
	if IndStep%100==0:
		IndRDFStat+=1.0
		for IndPartA in range(0,NumParticles-1):
			for IndPartB in range(IndPartA+1,NumParticles):
				DelX=Coord_X[IndPartA]-Coord_X[IndPartB]
				DelY=Coord_Y[IndPartA]-Coord_Y[IndPartB]
				DelZ=Coord_Z[IndPartA]-Coord_Z[IndPartB]

				#Apply boundary conditions#
				DelX-=Boxlength*int(DelX/Boxlength)
				DelY-=Boxlength*int(DelY/Boxlength)
				DelZ-=Boxlength*int(DelZ/Boxlength)

				DisPart=ma.sqrt(pow(DelX,2)+pow(DelY,2)+pow(DelZ,2))

				#Calculate in which bin this interaction is in#
				if DisPart<(0.5*Boxlength):
					RDFDis[int(DisPart/DeltaRDFGap)]+=2.0

	#Sample diffusion coefficient#
	time+=1
	if time%FreSeT0==0:
		#New time origin: store the positions/velocities; the current velocities#
		#are Velocities[i] and the current positions are PositionsNONPDB[i].#
		#question: why do you have to be careful with Pbc ?#
		#make sure to study algorithm 8 (page 91) of Frenkel/Smit#
		#before you start to make modifications note that most variable names are #
		#different here then in Frenkel/Smit. in this Way, you will have to think more...#
		t0Ind+=1
		t0Counter=(t0Ind-1)%MaxNumT0+1
		t0time[t0Counter]=IndStep
		for IndPart in range(0,NumParticles):
			Vxt0[IndPart][t0Counter]=Velocities_X[IndPart]
			Vyt0[IndPart][t0Counter]=Velocities_Y[IndPart]
			Vzt0[IndPart][t0Counter]=Velocities_Z[IndPart]

			Rx0[IndPart][t0Counter]=Coord_XNonPDB[IndPart]
			Ry0[IndPart][t0Counter]=Coord_YNonPDB[IndPart]
			Rz0[IndPart][t0Counter]=Coord_ZNonPDB[IndPart]

		t0Record=MaxNumT0
		if t0Ind<t0Record:
			t0Record=t0Ind
		for IndTime in range(1,t0Record):
			CorrelTime=IndStep-t0time[IndTime]+1
			if CorrelTime < MaxNumTS:
				SampleCounter[CorrelTime]+=1
				for IndPartCal in range(0,NumParticles):
					Vacf[CorrelTime]+=(Velocities_X[IndPartCal]*Vxt0[IndPartCal][IndTime]+
						Velocities_Y[IndPartCal]*Vyt0[IndPartCal][IndTime]+
						Velocities_Z[IndPartCal]*Vzt0[IndPartCal][IndTime])

					R2[CorrelTime]+=(pow(Coord_XNonPDB[IndPartCal]-Rx0[IndPartCal][IndTime],2)+
						pow(Coord_YNonPDB[IndPartCal]-Ry0[IndPartCal][IndTime],2)+
						pow(Coord_ZNonPDB[IndPartCal]-Rz0[IndPartCal][IndTime],2))
fMDTraj.close()
fEnD.close()
#################################################################################################
#################################################################################################
#################################################################################################
#				 	MD simulation done					#
#################################################################################################
#################################################################################################
#################################################################################################


#################################################
#	Write down final results		#
#################################################

if ParaAv[5]>0.5:	#Make sure MD run correctly#
	ParaAv[5]=1.0/ParaAv[5]
	for IndPara in range(0,5):
		ParaAv[IndPara]*=ParaAv[5]

	print('Average Temperature : %lf' % ParaAv[0],file=fMDSim)
	print('Average Pressure : %lf' % ParaAv[1],file=fMDSim)
	print('Average UKinetic : %lf' % ParaAv[2],file=fMDSim)
	print('Average UPotential : %lf' % ParaAv[3],file=fMDSim)
	print('Average UTotal : %lf' % ParaAv[4],file=fMDSim)
	#Print the average energy drift
	print('Average Delta Energy : %lf' % (dUtotAll*ParaAv[5]),file=fMDSim)
fMDSim.close()

#RDF results#
for IndRDF in range(0,MaxNumRDF-1):
	DisPart=float((4.0*M_PI/3.0)*(NumParticles/pow(Boxlength,3.0))*pow(DeltaRDFGap,3.0)*(pow(IndRDF+1,3.0)-pow(IndPart,3.0)))+0.001
	print('RDF value %f %f' % ((IndRDF+0.5)*DeltaRDFGap,RDFDis[IndRDF]/(IndRDFStat*NumParticles*DisPart)) , file=fMDRDF)
fMDRDF.close()

#####################################################
#Diffusion coefficient								#
#####################################################
CumIntegration=0.0
for IndTimeCorr in range(0,MaxNumTS-1):
	if SampleCounter[IndTimeCorr]>0:
		Vacf[IndTimeCorr]/=np.double(NumParticles*SampleCounter[IndTimeCorr])
		R2[IndTimeCorr]/=np.double(NumParticles*SampleCounter[IndTimeCorr])
	else:
		Vacf[IndTimeCorr]=0.0
		R2[IndTimeCorr]=0.0
	CumIntegration+=Tstep*Vacf[IndTimeCorr]/3.0
	print('%lf %lf %lf' % ((IndTimeCorr+1)*Tstep, Vacf[IndTimeCorr], CumIntegration) , file=fMDSimVACF)

	if IndTimeCorr>0:
		print('%lf %lf %lf' % ((IndTimeCorr+1)*Tstep, R2[IndTimeCorr], R2[IndTimeCorr]/(6.0*(IndTimeCorr+1)
			*Tstep)), file=fMDSimMSD)
	if IndTimeCorr==0:
		print('%lf %lf %lf' % ((IndTimeCorr+1)*Tstep, R2[IndTimeCorr], 0.0), file=fMDSimMSD)

fMDSimVACF.close()
fMDSimMSD.close()

