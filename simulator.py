from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from Generic.filedialogs import load_filename
from os.path import splitext, basename, dirname

class Particle:

    '''
    Class to store the properties of the particle such as position, vel

    We also define how the particle interacts with other particles and the boundary.
    '''
    num_particles = 0
    time_step=0
    next_id = 0

    def __init__(self, pos, vel, radius=1, mass=1, colour=(255,0,0)):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.radius = radius
        self.mass = mass
        self.colour=colour
        self.id = Particle.next_id
        Particle.next_id += 1
        Particle.num_particles += 1


    def update_pos(self):
        self.pos = self.pos + self.vel*Particle.time_step


    def update_vel(self, mu=0.1, force = 1):
        self.vel = (1 - mu) * self.vel + np.random.normal(loc=0.0, scale=0.5, size=np.shape(self.vel))#(force/self.mass)*Particle.time_step


    def update_property(self):
        pass



class Scene:
    '''
    Class to define the boundary conditions

    Here we define the nature of the boundary periodic or not
    the shape of the boundary and its location. The interaction with
    the boundary is set in the Particle class. If the boundary moves this
    is also defined here.
    '''
    def __init__(self, filename = None):





        self.grid_dict = {}

    def check_particle_boundary(self):
        pass


    def assign_balls_grid(self, nballs):

        for item in range(nballs):
            # Work out box coordinates
            i = int(math.floor((ball[item].pos.x + boxDim / 2) / boxDim))
            j = int(math.floor((ball[item].pos.y + boxDim / 2) / boxDim))
            ball[item].box = (i, j)  # Store box coordinates
            if (i, j) not in self.grid_dict:
                self.grid_dict[(i, j)] = []
            self.grid_dict[(i, j)].append(item)







class ParticleSimulator:
    '''
    Class to handle the laws of motion of the particle
    '''
    def __init__(self,file_handle, particles, time_step):
        self.f = file_handle
        self.time_step = time_step
        self.particles=particles



    def evolve(self, dt):

        nsteps = int(dt / self.time_step)
        #Particle.time_step = self.time_step
        # Loop order is changed
        for i in range(nsteps):
            [particle.update_pos() for particle in self.particles]
            [particle.update_vel() for particle in self.particles]
            #self.particles.update_property(self.timestep)


    def store_particle_data(self, particles, format = 'xyz'):

            self.f.write(str(Particle.num_particles)+'\n\n')
            [self.f.write(str(particle.id) + ' '
                     + str(particle.pos[0]) + ' ' + str(particle.pos[1]) + ' ' + str(particle.pos[2]) + ' '
                     + str(particle.vel[0]) + ' ' + str(particle.vel[1]) + ' ' + str(particle.vel[2]) + ' '
                     + str(particle.mass) + ' '
                     + str(particle.radius) + ' '
                     + str(particle.colour[0]) + ' ' + str(particle.colour[1]) + ' ' + str(particle.colour[2])
                     + '\n') for particle in particles]










if __name__ == '__main__':
    '''Use ipython to profile. From ipython command line:
    %load_ext line_profiler
    %load_ext memory_profiler
    from simulator import benchmark, benchmark_memory
    %timeit benchmark()
    %prun benchmark()
    %lprun -f ParticleSimulator.evolve benchmark()
    %mprun -f benchmark_memory benchmark_memory()
    '''

    #Define particles
    n_particles = 300
    particles = [Particle([0 , 0, 0], [0,0,0]) for i in range(n_particles)]
    #Define simulation boundary
    scene = Scene()
    #Data dump every dt
    dt = 1
    time_step = 0.001
    t_sim = 10


    filename_sim = load_filename(directory='/home/mike/Documents/particle_sim/init_sim/',
                                      file_filter='*.sim_setup')
    filename_scene = load_filename(directory='/home/mike/Documents/particle_sim/init_sim/', file_filter='*.scene')





    filename_op = dirname(filename_sim) + '/../sim_output/' + splitext(basename(filename_sim))[0] + '.sim_result'
    with open(filename_op, "a+") as f:
        # Create simulation
        simulator = ParticleSimulator(f,particles,time_step)
        simulator.store_particle_data(particles)
        for i in range(int(t_sim/dt)):
            simulator.evolve(dt)
            simulator.store_particle_data(particles)
