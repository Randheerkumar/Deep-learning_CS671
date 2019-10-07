############ Without Computational Graph ################
import numpy as np
import os
import timeit

#input path 
path="q2_input/"

#output path
output="q2_output_without_graph/"

#loading the masses,velocities and positions of all the particles
m=np.load(path+"masses.npy")
x=np.load(path+"positions.npy")
v=np.load(path+"velocities.npy")

#the constrain used in the problem
n=m.shape[0]
g=667000.0
t=0.0001
thres=0.1

#this function check that a pair of object exits having distance betwen them is less than threshold
def check_pair(x,threshold):
	n=x.shape[0]
	for i in range(n):
		for j in range(n):
			if i!=j:
				dist=pow((pow(x[j][0]-x[i][0],2)+pow(x[j][1]-x[i][1],2)),0.5)
				if dist<= threshold:
					return 1,i,j
	return 0,i,j			


#variable to store final velocities and distances	
x_final=np.zeros((n,2))
v_final=np.zeros((n,2))
acc=np.zeros((n,2))

# for i in range(n):
# 	print(i,x[i])

a=0
k=0

start_time = timeit.default_timer()

#here we calculate the final positions and velocities of particles when distance between any pair is less than threshold
while a!=1:
	for i in range(n):
		ax=0;ay=0;
		for j in range(n):
			if i!=j:
				temp=pow(pow(x[j][0]-x[i][0],2)+pow(x[j][1]-x[i][1],2),1.5)
				#print(i,j,temp)
				ax+=(g*m[j][0]*((x[j][0]-x[i][0]))/temp)
				ay+=(g*m[j][0]*((x[j][1]-x[i][1]))/temp)	
        
		acc[i][0]=ax
		acc[i][1]=ay
	x=x+v*t+(0.5)*acc*t*t	
	v=v+acc*t

	a,b,c=check_pair(x,thres)
	k+=1
	#print(k)
	if a==1:
		#print(k,b,c)
		np.save(output+"positions.npy",x)
		np.save(output+"velocities.npy",v)
		break;

elapsed = timeit.default_timer() - start_time
print("time taken to execute the code is ",elapsed,"seconds")
print("In",k,"th iteration distance between ",b,"th and ",c,"th particle becomes less than ",thres)

# for i in range(n):
# 	print(i,x[i])