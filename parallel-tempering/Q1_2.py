#!/usr/bin/env python

from random import *
import numpy as np
import math as ma
import random

#Set initial parameters#
Numstep=1000000	#Number of trial steps for each system#
Rho=2.5				#Density of all systems, i.e. N/(L*L)#
ProTrialEx=0.2		#Probability for swap states between two systems#
NumSystem=6			#Number of systems#
NumParticle=3 	#Number of particles in each system#
Temp=[0.05,0.025,0.01,0.005,0.0025,0.001]	#Temperature distributions for systems#
Beta=np.zeros(NumSystem,dtype=np.float)	#Beta for each system#
Epilson=1.0			#Reduced parameter value for calculating potential energy#
Displacement=0.05	#Max displacement of particle#
Rcut=1.0			#Cut-off radius for calculating potential energy#
NumRecordStep=1000	#Number of steps to record coordinate_X of 1st particle/energy#

#Coordinate of particles#
Coord_X=np.zeros((NumParticle,NumSystem),dtype=np.double)		#Coordinate X of Particle#
Coord_Y=np.zeros((NumParticle,NumSystem),dtype=np.double)		#Coordinate Y of Particle#
#Energy of system#
EnergyRecord=np.zeros(NumSystem,dtype=np.double)					#Energy of each system#


#Record coordinate_X of 1st particle# 
Coord_XSum=np.zeros(NumSystem,dtype=np.double)					#Summation of coordinate_X of 1st Particle#
Coord_XCount=np.zeros(NumSystem,dtype=np.double)					#Count times to add coordinate X of 1st Particle#
#Record energy 
EnergyRecordSum=np.zeros(NumSystem,dtype=np.double)				#Summarized energy of all systems#
EnergyRecordCount=np.zeros(NumSystem,dtype=np.double)			#Times to add energy for each system#
#Record accepted trial move in each system#
ParticleMoveSum=np.zeros(NumSystem,dtype=np.double)				#Total trial moves in system#
ParticleMoveCount=np.zeros(NumSystem,dtype=np.double)				#Accepted trial moves in system#
#Record swap times between systems#
SystemSwapSum=np.zeros(NumSystem,dtype=np.double)				#Total swap trials between systems#
SystemSwapCount=np.zeros(NumSystem,dtype=np.double)				#Sueccess swap trials between systems#


#Rho=NumParticle/(L*L)  ==>    L=sqrt(NumParticle/Rho)#
SquareLength=np.double(ma.sqrt(NumParticle/Rho))	#Calculate side length of square#
#SquareLength=1.86
LatticeGap=SquareLength/(ma.sqrt(NumParticle)+3.0)	#Artificial gap between two particles#
print('Latticegap %f' % LatticeGap)

#Generate beta#
for NumTemp in range(0,NumSystem):	
	Beta[NumTemp]=1.0/Temp[NumTemp]

#Generate initial Distribution of particles in each systems#
#Generate lattice distribution of particles for each system#
ParticleSum=0	#Record index value of particles along initialization#
MaxInd=int(ma.sqrt(NumParticle))
print('Initial number lattice %d' % MaxInd)
for IndX in range(1,MaxInd+1):
	for IndY in range(1,MaxInd+1):
		if ParticleSum<NumParticle:
			#Generate coordinates for same particle in different systems#
			for IndSystem in range(0,NumSystem):
				Coord_X[ParticleSum][IndSystem]=(IndX+1)*LatticeGap
				Coord_Y[ParticleSum][IndSystem]=(IndY+1)*LatticeGap
			ParticleSum+=1
#Initial coordinates of particles done#

#Initialization#
for IndOfSys in range(0,NumSystem):
	EnergyRecord[IndOfSys]=0.0
	ParticleMoveSum[IndOfSys]=1.0
	ParticleMoveCount[IndOfSys]=0.0
	SystemSwapSum[IndOfSys]=1.0
	SystemSwapCount[IndOfSys]=0.0
	EnergyRecordSum[IndOfSys]=1.0
	EnergyRecordCount[IndOfSys]=0.0
	Coord_XSum[IndOfSys]=1.0
	Coord_XCount[IndOfSys]=0.0

#Calculate initial energy#
for IndOfSys in range(0,NumSystem):
	for IndOfParticleA in range(0,NumParticle-1):
		for IndOfParticleB in range(IndOfParticleA+1,NumParticle):
			PartDis=np.double(ma.sqrt(pow(Coord_X[IndOfParticleA][IndOfSys]-Coord_X[IndOfParticleB][IndOfSys],2)+
				pow(Coord_Y[IndOfParticleA][IndOfSys]-Coord_Y[IndOfParticleB][IndOfSys],2)))
			if PartDis<=Rcut:
				EnergyRecord[IndOfSys]+=Epilson*pow(PartDis-1,2)
	print('Initial Energy %d' % EnergyRecord[IndOfSys])

#Record distribution of coordinateX and energy#
MaxNumBin=1000													#Maximum number of bins record coordinate/energy#
MaxEnergy=10.0*Epilson											#Maximum energy of system#
EnergyBinGap=MaxNumBin/MaxEnergy								#/energy gap between two bins#
CoordBinGap=MaxNumBin/SquareLength								#/coordinate gap between two bins#
print('Energy Coordinate Gap %f %f' % (EnergyBinGap,CoordBinGap))
CoordXDis=np.zeros((MaxNumBin,NumSystem),dtype=np.float)		#Record distribution of coordinate X for 1st Particle#
EnergyDis=np.zeros((MaxNumBin,NumSystem),dtype=np.float)		#Record distribution of energy for all systems#

#Initilization of CoordXDis/EnergyDis#
for IndOfBin in range(0,MaxNumBin):
	for IndOfSys in range(0,NumSystem):
		CoordXDis[IndOfBin][IndOfSys]=0.0
		EnergyDis[IndOfBin][IndOfSys]=0.0
DisTime=0.0		#Record times to cal distribution of coordinate/energy#

#Start sampling#
for IndNumStep	in range(0,Numstep): 
	if IndNumStep%1000==0:
		#Visualize the progress of simulation#
		print('\r Current progress:%.3f%%' % (IndNumStep*100/Numstep))
#Exchange states between two systems when random < Pte(0.2)#
	Pswitch=1		#Probability to switch from state exchange to particle move#
	if ((Pswitch<ProTrialEx) and (NumSystem>1)):
		SetTempA=int(random.random()*(NumSystem-1))	#Select random system (<=NumSystem-1) for exchange states#
		SetTempB=SetTempA+1
		SystemSwapSum[SetTempA]+=1.0
		if random.random()<ma.exp((Beta[SetTempA]-Beta[SetTempB])*(EnergyRecord[SetTempA]-EnergyRecord[SetTempB])):
			for IndExParticle in range(0,NumParticle):
				Xmid=Coord_X[IndExParticle][SetTempA]
				Coord_X[IndExParticle][SetTempA]=Coord_X[IndExParticle][SetTempB]
				Coord_X[IndExParticle][SetTempB]=Xmid

				Ymid=Coord_Y[IndExParticle][SetTempA]
				Coord_Y[IndExParticle][SetTempA]=Coord_Y[IndExParticle][SetTempB]
				Coord_Y[IndExParticle][SetTempB]=Ymid						

			#Exchange energy#
			EnergyMid=EnergyRecord[SetTempA]
			EnergyRecord[SetTempA]=EnergyRecord[SetTempB]
			EnergyRecord[SetTempB]=EnergyMid
			#States Exchange down#
			SystemSwapCount[SetTempA]+=1.0
	#Else, trial move of particle in random system#
	if Pswitch>=ProTrialEx:
		SetTempA=int(random.random()*NumSystem)		#Random choose system#
		SetPartA=int(random.random()*NumParticle)	#Random choose particle#

		#Record old coordinates#
		CoordXOld=Coord_X[SetPartA][SetTempA]
		CoordYOld=Coord_Y[SetPartA][SetTempA]

		#Generate new coordinates#
		CoordXNew=0.0
		CoordYNew=0.0
		while( (CoordXNew<=0) or (CoordXNew>(SquareLength))):
			CoordXNew=Coord_X[SetPartA][SetTempA]+(random.random()-0.5)*Displacement
		while( (CoordYNew<=0) or (CoordYNew>(SquareLength))):
			CoordYNew=Coord_Y[SetPartA][SetTempA]+(random.random()-0.5)*Displacement

		Coord_X[SetPartA][SetTempA]=CoordXNew
		Coord_Y[SetPartA][SetTempA]=CoordYNew

		ParticleMoveSum[SetTempA]+=1.0

		EnergyNew=0.0
		for IndNumPartA in range(0,NumParticle-1):
			for IndNumPartB in range(IndNumPartA+1,NumParticle):
				PartDis=np.double(ma.sqrt(pow(Coord_X[IndNumPartA][SetTempA]-Coord_X[IndNumPartB][SetTempA],2)
							+pow(Coord_Y[IndNumPartA][SetTempA]-Coord_Y[IndNumPartB][SetTempA],2)))
				if PartDis<=Rcut:
					EnergyNew+=Epilson*pow(PartDis-1.0,2)
		Pmvtrial=random.random()
		if Pmvtrial<ma.exp((-1.0)*Beta[SetTempA]*(EnergyNew-EnergyRecord[SetTempA])):
			EnergyRecord[SetTempA]=EnergyNew
			ParticleMoveCount[SetTempA]+=1.0
			#Particle move trial done#
		else:
			Coord_X[SetPartA][SetTempA]=CoordXOld
			Coord_Y[SetPartA][SetTempA]=CoordYOld

	#Record coordinate_X of 1st particle and Energy distribution in each system when Nstep%10000==0#
	if IndNumStep%10==0:
		DisTime+=1.0
		for IndOfSys in range(0,NumSystem):
			#Record energy of each system and prepared to calculate average energy#
			EnergyRecordSum[IndOfSys]+=EnergyRecord[IndOfSys]
			EnergyRecordCount[IndOfSys]+=1.0
			#Record coordinate of 1st particle in each system and prepared to calculate average coordinate#
			Coord_XSum[IndOfSys]+=(Coord_X[0][IndOfSys]/SquareLength)
			Coord_XCount[IndOfSys]+=1.0

			IndEnergy=int(EnergyRecord[IndOfSys]*EnergyBinGap)
			if IndEnergy<MaxNumBin:
				EnergyDis[IndEnergy][IndOfSys]+=1.0
			#Record energy distribution#

			IndCoordPart=int(Coord_X[0][IndOfSys]*CoordBinGap)
			if IndCoordPart<MaxNumBin:
				CoordXDis[IndCoordPart][IndOfSys]+=1.0
			#Record coordinate distribution#		


DisData='DistributionData.dat'
fDis=open(DisData,'w+')
for IndNumDis in range(0,MaxNumBin):
	for IndOfSys in range(0,NumSystem):
		if IndOfSys==0:
			line=' DisIndex '+str(float(IndNumDis*1.0))+' CoordX '+str(float(CoordXDis[IndNumDis][IndOfSys]/DisTime)
					)+'EnergyVa'+str(float(IndNumDis/EnergyBinGap))+' Energy '+ str(float(EnergyDis[IndNumDis][IndOfSys]/DisTime))
			print(line,end='',file=fDis)
		if IndOfSys>0:
			line=' CoordX '+str(float(CoordXDis[IndNumDis][IndOfSys]/DisTime)
					)+' Energy '+ str(float(EnergyDis[IndNumDis][IndOfSys]/DisTime))
			print(line,end='',file=fDis)
	if IndOfSys==(NumSystem-1):
		print(' End',file=fDis)
fDis.close()

AveData='AverageData.dat'
f=open(AveData,'w+')
for IndNumSys in range(0,NumSystem):
	#Write averge coordinate_X of 1st particle in each system#
	#Write average acceptance probability in each system# 
	#Write average swap acceptance probability between i/i+1 systems#
	lines=' AverageCoordX '+str(float(Coord_XSum[IndNumSys]/Coord_XCount[IndNumSys])
			)+' AverageEnergy '+str(float(EnergyRecordSum[IndNumSys]/EnergyRecordCount[IndNumSys])
			)+' AverageAcceptance '+str(float(ParticleMoveCount[IndNumSys]/ParticleMoveSum[IndNumSys])
			)+' AverageSwap '+str(float(SystemSwapCount[IndNumSys]/SystemSwapSum[IndNumSys]))
	print(lines,file=f)
f.close()
