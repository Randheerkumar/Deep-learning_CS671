import numpy as np
import tensorflow as tf
import os
import timeit

#input file path
path="q2_input/"

#output file path
output="q2_output_with_graph/"

#loading the masses,velocities and positions of all the particles
m1=np.load(path+"masses.npy")
x1=np.load(path+"positions.npy")
v1=np.load(path+"velocities.npy")

#constraint used in problem
n=m1.shape[0]
g=667000.0
t=0.0001
thres=0.1

#various place holder that will take values later on
x=tf.placeholder(tf.float64,[n,2],name="x")
m=tf.placeholder(tf.float64,[n,1],name="m")
v=tf.placeholder(tf.float64,[n,2],name="v")


# for i in range(n):
#     print(i,x1[i])


#this function check that a pair of object exits having distance betwen them is less than threshold    
def check_pair(x,threshold):
    n=x.shape[0]
    for i in range(n):
        for j in range(n):
            if i!=j:
                dist=pow(pow(x[j][0]-x[i][0],2)+pow(x[j][1]-x[i][1],2),0.5)
                if dist<= threshold:
                    return 1,i,j
    return 0,i,j


#here we are makeing the computational graph
z1=x[:,0]-tf.transpose([x[:,0]])
z2=x[:,1]-tf.transpose([x[:,1]])
X=(tf.stack([z1, z2],0))
X_square=X*X
X_square_sum=tf.reduce_sum(X_square, 0,name="reduce_sum_along_0th_axis")
X_square_sum_pow=tf.pow(X_square_sum,1.5,name="raise_elemet_to_power_1.5")
comparison = tf.equal(X_square_sum_pow, tf.constant( 0,tf.float64))
X_square_sum_pow_new= tf.where(comparison,tf.subtract(X_square_sum_pow,1),X_square_sum_pow)
X_new=X/X_square_sum_pow_new
X_m=X_new*tf.transpose(m)
X_sum=tf.reduce_sum(X_m,2,name="reduce_sum_along_2nd_axis")
a=tf.linalg.transpose(X_sum,name="transpose")*g

#updating the velocities and positions
v_final=v+a*t
x_final=x+v*t+(0.5)*a*t*t

start_time = timeit.default_timer()

#here we start the session to compute the final velocities and positions of particles
with tf.Session() as sess:
    writer = tf.summary.FileWriter("output_log", sess.graph)
    a1=0;j=0
    while(a1!=1):
        fd={x:x1,v:v1,m:m1}
        v_f=sess.run(v_final,feed_dict=fd)
        x_f=sess.run(x_final,feed_dict=fd)

        v1=v_f
        x1=x_f
        a1,b1,c1=check_pair(x1,thres)
        j+=1
        if a1==1:
            #print(j,b1,c1)
            np.save(output+"positions.npy",x1)
            np.save(output+"velocities.npy",v1)
            break;
    # for i in range(n):
    #     print(i,x1[i])
    # print(v_f)
    writer.close()
    
elapsed = timeit.default_timer() - start_time
print("time taken to execute this code is",elapsed,"seconds")
print("In",j,"th iteration distance between ",b1,"th and ",c1,"th particle becomes less than ",thres)		