from __future__ import division
#import tkinter as tk
import tkFileDialog
import numpy as np
import pandas as pd
import shelve
#import os
import itertools
from vpython import*

#import pyximport; pyximport.install(pyximport=True)

#I think numpy and visual have a function called rate so make sure numpy is imported before visual.
import math

#import cythonSimFns
#if 'posX' in locals():
#    del(posX,posY,vel,velx)


##############################################################################
#Initialise particle grid
##############################################################################

#gridDict = {}#Dictionary specifying which particle index (values) is in each box (key). Box coordinates are specified by tuple of x,y coordinates


def assignBalls2Box(NBalls):
    #wipe previous vals
    gridDict =  {}
    for item in range(NBalls):
        #Work out box coordinates
        i = int(math.floor((ball[item].pos.x + boxDim/2)/boxDim))
        j = int(math.floor((ball[item].pos.y + boxDim/2)/boxDim))
        ball[item].box = (i,j)#Store box coordinates
        if (i,j) not in gridDict:
            gridDict[(i,j)] = []
        gridDict[(i,j)].append(item)
    return gridDict
            
    

def identifyNeighbouringParticles(item):
    vals= []    
    i,j=ball[item].box
    
    if item in gridDict[(i,j)]:         
        #Remove item from the dictionary so we don't need to recheck it when calculating for something else
        gridDict[(i,j)] = list(set(gridDict[i,j]) - set([item]))
        
        if boundarytype =='hard': 
            #Look at neighbouring boxes
            keys = [(i+1,j+1),(i,j+1),(i-1,j+1),(i+1,j),(i,j),(i-1,j),(i+1,j-1),(i,j-1),(i-1,j-1)]
        
        elif boundarytype == 'periodic':        
            #Case where particles are not at a boundary
            if i**2 < (iboundary)**2 and j**2 < (jboundary)**2 :
                keys = [(i+1,j+1),(i,j+1),(i-1,j+1),(i+1,j),(i,j),(i-1,j),(i+1,j-1),(i,j-1),(i-1,j-1)]
            #Case where particles are at x boundary but not y
            elif i**2 == iboundary**2 and j**2 < (jboundary-1)**2:
                keys = [(-i,j+1),(i,j+1),(i-1,j+1),(-i,j),(i,j),(i-1,j),(-i,j-1),(i,j-1),(i-1,j-1)]
            #Case where particles are at y boundary but not x
            elif i**2 < (iboundary-1)**2 and j**2 == jboundary**2:
                keys = [(i+1,-j),(i,-j),(i-1,-j),(i+1,j),(i,j),(i-1,j),(i+1,j-1),(i,j-1),(i-1,j-1)]
            #Cases where particles are in the corners.
            elif i == iboundary and j == jboundary:
                keys = [(i,j),(i,j-1), (i-1,j-1),(i-1,j),(i,-j),(i-1,-j),(-i,-j),(-i,j),(-i,j-1)]
            elif i == iboundary  and j == -jboundary:
                keys = [(i,j),(i,j+1), (i-1,j+1),(i-1,j),(i,-j),(i-1,-j),(-i,-j),(-i,j),(-i,j+1)]
            elif i == -iboundary and j == jboundary:
                keys = [(i,j),(i,j-1), (i+1,j-1),(i+1,j),(i,-j),(i+1,-j),(-i,-j),(-i,j),(-i,j-1)]
            elif i == -iboundary and j == -jboundary:
                keys = [(i,j),(i,j+1), (i+1,j+1),(i+1,j),(i,-j),(i+1,-j),(-i,-j),(-i,j),(-i,j+1)]
            else:
                keys = [1]
            
        
        for key in keys: 
            if key in gridDict:
                #MUST BE SOME SPEED UP TO BE DONE HERE
                #Find particle indices
                vals.append(gridDict[key])  
        
        #Return list of all balls in same or neighbouring boxes
        return list(itertools.chain.from_iterable(vals)) 
    else:
        return 'notPresent'               

###############################################################################
# Set up the classes
###############################################################################

class ballObject:
    def __init__(self,radBall,massBall,index):
        self.rad = radBall
        self.mass = massBall
        self.pos,self.vel = self.createInitialPos()
        i = int(math.floor((self.pos.x + boxDim/2)/boxDim))
        j = int(math.floor((self.pos.y + boxDim/2)/boxDim))
        self.box = (i,j)
        self.index = index
        

    def createInitialPos(self):       
        if boundarytype == 'hard':  
            while True:
                xval = np.random.uniform(-lims+self.rad,lims-self.rad)
                yval = np.random.uniform(-lims+self.rad,lims-self.rad)
                if (xval**2 + yval**2) <= ((lims-self.rad)**2):
                    return (vector(xval,yval,0),vector(0,0,0))
        elif boundarytype == 'periodic':            
            xval = np.random.uniform(-SqEdgeDim/2+self.rad,SqEdgeDim/2-self.rad)
            yval = np.random.uniform(-SqEdgeDim/2+self.rad,SqEdgeDim/2-self.rad)
            return (vector(xval,yval,0),vector(0,0,0))
                

    def collideWallPeriodic(self):
        collidewall = False
        pos = self.pos
        testpos = pos + self.vel*dt
        if (testpos.x**2 < (SqEdgeDim/2)**2) and (testpos.y**2 < (SqEdgeDim/2)**2):
            #Particle does not collide with wall            
            return collidewall
        else:
            collidewall = True
            #Calculate position on boundary of collision.
            if testpos.x >= SqEdgeDim/2:
                self.pos = testpos - vector(SqEdgeDim,0,0)
            if testpos.x <= -SqEdgeDim/2:
                self.pos = testpos + vector(SqEdgeDim,0,0)
            if testpos.y >= SqEdgeDim/2:
                self.pos = testpos - vector(0,SqEdgeDim,0)
            if testpos.y <= -SqEdgeDim/2:
                self.pos = testpos + vector(0,SqEdgeDim,0)
            return collidewall
            
    def collideWallHard(self):  
        
        collidewall = False
        pos = self.pos
        vel = self.vel
        radval = inner_ringradius - Radii[self.index]
        if mag(pos + vel*dt) < radval:
            #Particle does not collide with wall            
            return collidewall
        else:
            collidewall = True
            #There is a very small (5 decimal places) precision dividing error which 
            #sometimes results in beads being a tiny amount over wall. We shift these beads
            #to prevent this causing a negative in the   
         
            #Where do we make contact with wall. Solve quadratic intersection of line with circle
            A  = (dt**2)*(vel.x**2 + vel.y**2)   # always positive     
            B = (2*pos.x*vel.x*dt) + (2*pos.y*vel.y*dt) #
            C = pos.x**2 + pos.y**2 - radval**2 #Always negative
            Calc = (B**2 - 4*A*C)
      
            if B**2 < 4*A*C:# means there is no intersection
                Calc=0
      
            #Since the bead that is hit must be going to hit the wall
            partsoln = Calc**0.5
            #The solution we want will be positive
            Solution1 = (-B + partsoln)/(2*A)
            
            part_dt = dt*Solution1
            #Something funny going on
            if part_dt >= dt:
                Solution1 = (-B - partsoln)/(2*A)
                part_dt = dt*Solution1            
            
            RelCollidePos = Solution1*vel*dt# dt changed to part_dt
           
            #Unit vector normal to edge where bead collides
            Normal = (RelCollidePos + pos)/mag(RelCollidePos+pos)
            Tangential = vector(Normal.y,-Normal.x,0)
            #Time between start of time step and collision with wall
            
 
            #Calc components radial and transverse
            Vrad = dot(vel,Normal)
            Vtheta = dot(vel,Tangential)       
    
            if (abs((Vrad)*(fcoeff_Boundary))) > abs(Vtheta):
                Vtheta = 0       
            else:
                if Vtheta > 0:
                    Vtheta = Vtheta -(abs((Vrad)*(fcoeff_Boundary)))
                else:
                    Vtheta = Vtheta+(abs((Vrad)*(fcoeff_Boundary)))   
     
            vel = -(Coeff_Rest_Boundary * (Vrad*Normal)) +(Vtheta*Tangential)
            self.vel=vel
            #Position after collision		
            PosAfter = pos + RelCollidePos + vel*(dt - part_dt)              
            self.pos = PosAfter
                    
                    
                
                #collision_array_wall[item] = 0
            
        return collidewall
        
    def collisionCalc(self,id):
        collideball = False
        vel = self.vel#item
        pos = self.pos
        vel_id = ball[id].vel#id
        pos_id = ball[id].pos
        
        #http://www.gamasutra.com/view/feature/131424/pool_hall_lessons_fast_accurate_.php?page=2
        #Nomenclature is taken from this website which contains diagrams explaining all the steps
        #Everything is done in the frame in which the bead being hit is stationary and then transformed back
         
        #http://www.gamasutra.com/view/feature/131424/pool_hall_lessons_fast_accurate_.php?page=2
        #Nomenclature is taken from this website which contains diagrams explaining all the steps
        #Everything is done in the frame in which the bead being hit is stationary and then transformed back
        #Keep track of whether bead has collided or not.
        #This is the relative velocity and position vector. It is the relative velocity of item with respect to a static id
        rv = vel - vel_id#Relative velocity vector
        C = -(pos - pos_id)#Relative positoin vector
        TooFar = (mag(C) - mag(rv)*dt ) # Distance travelled in timestep - distance between centres. Early exit strategy
        SumRadii = Radii[self.index] + Radii[id]
            
        if TooFar < SumRadii:
            #Are objects moving towards one another. No < 0
            if dot(C,rv) >= 0:
                #If closest they get > 2 * rad they don't hit
                Nv = rv/mag(rv)#Normalised velocity vector
                D=dot(Nv,C)
                F = mag(C)**2 - D**2
                diamsq = (SumRadii)**2
                #If false they don't touch
                if F  <= diamsq:
                    T = diamsq - F
                    Distance = D - T**0.5
                    
                    Step = mag(rv)*dt
                    if Distance < Step:
                        collideball = True
                        #If you get in here you have definitely got a collision. The Bead moves RelMove2Touch along the relative velocity vector before the edges touch.
                        #RelMove2touch = N*Distance #This is a relative movement vector to bring edges of particles to touch.
                        part_dt = Distance / mag(rv)#Time taken for beads to touch
                        pos_item_attouch = pos + vel*part_dt
                        pos_id_attouch = pos_id + vel_id*part_dt
                        C2Cattouch = pos_id_attouch - pos_item_attouch
                        #velocity components normal to collision
                        #http://gamedevelopment.tutsplus.com/tutorials/how-to-create-a-custom-2d-physics-engine-friction-scene-and-jump-table--gamedev-7756
                        Normal_unitvector =  -C2Cattouch/mag(C2Cattouch)
                        #Va = vel
                        #Vb = vel_id
                        #rv = vel - vel_id
                        rvn = dot(rv,Normal_unitvector)
                        Tangential_unitvector = vector(Normal_unitvector.y,-Normal_unitvector.x,0)
                        rvt = dot(rv,Tangential_unitvector)
                        j = -(1+Coeff_Rest_Bead)/(2/mass)                                                                                                             
                        jn = j*(rvn)
                        jt = j*(rvt)
                        
                        Vpa = vel + jn*Normal_unitvector/mass 
                        Vpb = vel_id - jn*Normal_unitvector/mass
                        
                        vpan = dot(Vpa,Normal_unitvector)
                        vpbn = dot(Vpb,Normal_unitvector)                                                      
                        
                        vpat = dot(Vpa,Tangential_unitvector)
                        vpbt = dot(Vpb,Tangential_unitvector)                        
                        
                        Van = dot(vel,Normal_unitvector)
                        Vbn = dot(vel_id,Normal_unitvector)
                        
                        if (abs((Van+ Vbn)*(fcoeff_Bead))) > abs(vpat):
                            fa = 0                             
                        else:
                            if vpat > 0:
                                fa = vpat-(abs((Van+ Vbn)*(fcoeff_Bead)))
                            else:
                                fa = vpat+(abs((Van+ Vbn)*(fcoeff_Bead)))
                        
                            
                            if (abs((Van+ Vbn)*(fcoeff_Bead))) > abs(vpbt) :
                                fb = 0                              
                            else:
                                if vpbt > 0:
                                    fb = vpbt-(abs((Van+ Vbn)*(fcoeff_Bead)))
                                else:
                                    fb = vpbt+(abs((Van+ Vbn)*(fcoeff_Bead)))
                        
                            
                            Vppa = dot(Vpa,Normal_unitvector)*Normal_unitvector + fa*Tangential_unitvector
                            Vppb = dot(Vpb,Normal_unitvector)*Normal_unitvector + fb*Tangential_unitvector                                                      
                            self.vel = Vppa
                            self.pos = pos_item_attouch + Vppa*(dt - part_dt)
                            ball[id].vel = Vppb
                            ball[id].pos = pos_id_attouch + Vppb*(dt - part_dt)
          
        return collideball							

    def collideBall(self,neighbours):
        #Neighbours is a list of indexes that are in neighbouring boxes. If the particle has already moved it returns False
        collidebead = False
        if neighbours == False:
            return True#This scenario means the item has already collided and had its position updated. We return true to prevent its positoin being updated again
        else:
            #If there are no neighbours this should skip otherwise it checks whether each bead has collided.
            for id in neighbours:
                #Calculate collision if true updates position of both beads
                collidebead = self.collisionCalc(id)
                #print(collidebead)            
        
                if collidebead:
                    
                    #Remove collided with bead from box of particles since we assume it can't
                    #collide with 2 particles in single timestep
                    gridDict[ball[id].box] = list(set(gridDict[ball[id].box]) - set([id]))
                    #returns true since collision occurred
                    #Remove colliding ball from gridDict        
                    gridDict[self.box] = list(set(gridDict[self.box]) - set([self.index])) 
                    #print(neighbours)
                                      
                    return collidebead	
        #Returns false since no collisions        
        return collidebead

    def updatePos(self):
        #print(self.index)
        neighbours = identifyNeighbouringParticles(self.index) 
        if neighbours == 'notPresent':
            pass
        else:
            if boundarytype == 'hard':
                collidewall = self.collideWallHard()
            elif boundarytype == 'periodic':
                collidewall = self.collideWallPeriodic()
             
            #Check for interparticle collisions. If collides move particle and collided with particle
            collideball = self.collideBall(neighbours)
            
            if (not collideball and not collidewall):
                #If no collisions update position
                self.pos = self.pos + self.vel*dt
                #Remove index from grid dictionary
                gridDict[self.box] = list(set(gridDict[self.box]) - set([self.index]))
	
    
        
        
    def updateVel(self):
        self.vel.x = (1-mu)*self.vel.x + np.random.normal(0,sigma)
        self.vel.y = (1-mu)*self.vel.y + np.random.normal(0,sigma)
        

###############################################################################
#Code to initialise balls
##############################################################################

##############################################################################################################################################
#Specify parameters of the simulation
#############################################################################################################################################
filename = filedialog.askopenfilename(defaultextension='.db',initialdir = '~/Documents/Data/ParticleSimData')
#initial_name = os.path.split(filename)[1]
#filenamesave = tkFileDialog.asksaveasfilename(defaultextension='.txt',initialdir = 'C:\Users\ppxnds\Documents\Python\Data', initialfile = [initial_name[0:-3] + '.txt'])

#filename = u'C:/Users/ppzmis/Documents/LocalPython/timetest.db'    
s = shelve.open(filename)


#Create Boundary
inner_ringradius = s['boundary']['inner_ringradius']
#inner_ringradius = 11.25
ringthickness = s['boundary']['ringthickness']
#make the internal ring dimension = ring radius
#ringradius = inner_ringradius + ringthickness
lims = inner_ringradius
fcoeff_Boundary = s['boundary']['fcoeff_boundary']#` # Friction coefficient for tangential velocity component at boundary
Coeff_Rest_Boundary = s['boundary']['coeff_boundary']#0.85 # Coefficient of restitution affects normal velocity component
#If the boundary condition is periodic we calculate the size of square to give same area as circle (this means the area fraction of particles remains the same)
Area = np.pi*inner_ringradius**2
SqEdgeDim = Area**0.5


#Bead Params
Radii = s['ball']['radii']
NumbBalls = s['ball']['numbballs']
fcoeff_Bead = s['ball']['fcoeff_bead']
Coeff_Rest_Bead = s['ball']['coeff_bead']
mu = s['ball']['visc']
mass = s['ball']['mass']

try:
    #Simulation params
    dt = s['simParams']['dt']#Timestep
    sigma =s['simParams']['sigma']
    runs = s['simParams']['runs']
    startStoring = s['simParams']['startstoring']
    boxDim = s['simParams']['boxDim']
    boundarytype = s['boundary']['boundarytype']
    
    #Only relevant to periodic boundary conditions specifies max coordinate of the square
    iboundary = int(math.floor((SqEdgeDim/2 + boxDim/2)/boxDim))
    jboundary = int(math.floor((SqEdgeDim/2 + boxDim/2)/boxDim))
except:
    'Check this is a simulation in the shelf file'
#File outputs two columns radial position and magnitude of velocity
OutputFile = s['output']['outputfile']
filenamebase = s['output']['filenameop']



###############################################################################
#Param definitions finished into main function
###############################################################################


'''
ball is a list of ballObjects which each has
properties:
    radius
    mass
    position
    velocity
    
methods:
    createInitialPos
    collisionWalls

'''
ball = [ballObject(Radii[item],mass,item) for item in range(NumbBalls)]
#Main Loop
cnt = 0
arrayBalls = range(NumbBalls)


'''
Define position and velocity arrays which will contain output from simulation 
and be written to file
'''
ovito_output = pd.DataFrame(index=np.arange(0, (NumbBalls+2)*(runs-startStoring)+1), columns=('particle_num','x', 'y','z','r','radius', 'vx','vy','vz','vr','vt','vmag') )
#position=np.zeros((NumbBalls*(runs-startStoring),4))
#velocity=np.zeros((NumbBalls*(runs-startStoring),6))
df = pd.DataFrame(index=np.arange(0, NumbBalls*(runs-startStoring)), columns=('particle_num','x', 'y','z','r','radius', 'vx','vy','vz','vr','vt','vmag'))


while cnt < runs:
    #Assign balls to particular box
    gridDict={}#Clear dictionary from previous iteration
    gridDict = assignBalls2Box(NumbBalls) # Add balls to each box. Checked
    #print(gridDict)

    #Update the velocities of all balls    
    [ball[item].updateVel() for item in arrayBalls]

    if cnt >= startStoring:
        ovito_output.loc[(cnt-startStoring)*(NumbBalls + 2)][:] = [NumbBalls,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        ovito_output.loc[(cnt-startStoring)*(NumbBalls + 2)+1][:] = [cnt-startStoring,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]   
    #Update the positions of the balls
    for item in arrayBalls: 
        ball[item].updatePos()
        
        if cnt >= startStoring:
            Vr = proj(ball[item].vel,ball[item].pos)#Radial vel vector
            magVth = mag(ball[item].vel - Vr)#tangential vel magnitude
            
            #Store x,y,r
            ovito_output.loc[(cnt-startStoring)*(NumbBalls + 2) +item+2][:] = [item, ball[item].pos.x, ball[item].pos.y, 0, mag(ball[item].pos),Radii[item],ball[item].vel.x, ball[item].vel.y, 0, mag(Vr), magVth, mag(ball[item].vel)]
            
            #position[(cnt-startStoring)*NumbBalls + item][:] = [item, ball[item].pos.x, ball[item].pos.y, 0]#, mag(ball[item].pos)]
            #Store vx,vy,vr,vth,magv
            #velocity[(cnt-startStoring)*NumbBalls + item][:] = [ball[item].vel.x, ball[item].vel.y, 0, mag(Vr), magVth, mag(ball[item].vel)]
            df.loc[(cnt-startStoring)*NumbBalls + item] = [item, ball[item].pos.x, ball[item].pos.y, 0, mag(ball[item].pos),Radii[item],ball[item].vel.x, ball[item].vel.y, 0, mag(Vr), magVth, mag(ball[item].vel)]
    
    print(cnt)  
    cnt = cnt+1
    

#print(np.shape(position))
#np.savetxt(filenamebase + '_position.xyz',position)
#np.savetxt(filenamebase + '_velocity.txt',velocity)
#df.to_hdf(filenamebase + 'hdffile','simdata')
#np.savetxt(filenamebase + '_ovito.xyz',ovito_output,delimiter='\t')

ovito_output.to_csv(filenamebase + '_ovito.xyz',sep='\t',header=False,index=False)
df.to_hdf(filenamebase + '_hdffile','fullsimdata')
s.close()

   
    #Do we write data line by line to file or dataframe or wait until the end
    










        