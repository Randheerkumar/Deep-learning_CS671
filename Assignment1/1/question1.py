#importing the requied libaries
import numpy as np
import cv2
import math
import os
import glob

#defining the parameters of the line
Length=[7,15]
Width=[1,3]
Color=[[0,0,255],[255,0,0]]
Angle=[0,15,30,45,60,75,90,105,120,135,150,165]


#this function generate the imgages with differnt variations
def generate_line():
	j=1
	for l in range(len(Length)):
		for w in range(len(Width)):
			for a in range(len(Angle)):
				for c in range(len(Color)):
					i=1
					directory="Image/"+"Class"+"_"+str(j)
					if not os.path.exists(directory):
						os.makedirs(directory)
					j+=1
					image_name=str(l)+"_"+str(w)+"_"+str(a)+"_"+str(c)+"_"
					s=(int)(l/2)+1
					num=0
					while(num<1000):
						for x in range(s,28-s):
							for y in range(s,28-s):
								image_name1=image_name+str(i)+".jpg"
								img = np.zeros((28,28,3), np.uint8)
								angle = Angle[a]
								length =Length[l]/2.0;
								x1=(int)(x-length * math.cos(math.radians(angle)))
								y1=(int)(y+length * math.sin(math.radians(angle)))
								x2=(int)(x + length * math.cos(math.radians(angle)))
								y2=(int)(y - length * math.sin(math.radians(angle)))

								img = cv2.line(img,(x1,y1),(x2,y2),(Color[c]),Width[w])

								cv2.imwrite(directory+"/"+image_name1,img)
								#print(i)
								i+=1
								num+=1
								if(num>=1000):
									break

							if(num>=1000):
								break
						if(num>=1000):
							break		



#this function make the frame of images
def generate_frame():
	folder="Image/"
	frame_folder="Frame_Image/"
	j=0
	for Class in sorted(os.listdir(folder)):
		i=0;k=0
		frame_folder1=frame_folder+"Class_"+str(j)
		if not os.path.exists(frame_folder1):
						os.makedirs(frame_folder1)
		for filename in sorted(glob.glob(folder+Class+'/*.jpg')):
			if i<90:
				if i%9==0:
					img0=cv2.imread(filename)
				elif i%9==1:
					img1=cv2.imread(filename)
					Img1=np.append(img0,img1,axis=0) 
				elif i%9==2:
					img2=cv2.imread(filename)
					Img1=np.append(Img1,img2,axis=0) 
				elif i%9==3: 
					img3=cv2.imread(filename)
				elif i%9==4:
					img4=cv2.imread(filename)
					Img2=np.append(img3,img4,axis=0) 
				elif i%9==5: 
					img5=cv2.imread(filename)
					Img2=np.append(Img2,img5,axis=0)  
				elif i%9==6: 
					img6=cv2.imread(filename)
				elif i%9==7: 
					img7=cv2.imread(filename)
					Img3=np.append(img6,img7,axis=0) 
				elif i%9==8:
					img8=cv2.imread(filename) 
					Img3=np.append(Img3,img8,axis=0)
					Img=np.append(Img1,Img2,axis=1)
					Img=np.append(Img,Img3,axis=1)
					cv2.imwrite(frame_folder1+"/"+str(k)+".png",Img) 
					k+=1            
				i+=1
		j+=1		



#this function make the video after joing the frames
def make_video():
	img_array = []			
	folder="/home/dilip/Documents/course/6thsem/DL/Assignment/Assignment1/Frame_Image/"
	size=(0,0)
	for Class in sorted(os.listdir(folder)):
		for filename in sorted(glob.glob(folder+Class+'/*.png')):
			img = cv2.imread(filename)
			#print(filename)
			height, width, layers = img.shape
			size = (width,height)
			img_array.append(img)

	out = cv2.VideoWriter('assignment1_1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2 , size)

	for i in range(len(img_array)):
	    out.write(img_array[i])
	out.release()
	

#calling different function
generate_line()
generate_frame()
make_video()
