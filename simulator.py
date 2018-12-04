from matplotlib import pyplot as plt
from matplotlib import animation
from random import uniform
import timeit

class Particle:
    '''
    Class to store the properties of the particle such as position, vel
    '''
    def __init__(self, x, y, mass=None, rad=None):
        self.x = x
        self.y = y
        self.mass = mass
        self.rad = rad
        self.ang_vel = ang_vel

class Boundary:
    '''
    Class to define the boundary conditions
    '''
    def __init__(self):
        pass





class ParticleSimulator:
    '''
    Class to handle the laws of motion of the particle
    '''
    #Slots reduces the memory footprint of Particle class by avoiding storing variables of an instance in an internal
    #dictionary. Drawback is it prevents addition of attributes no in __slots__
    __slots__ = ('x','y','ang_vel')

    def __init__(self, particles):
        self.particles = particles


    def evolve(self, dt):
        timestep = 0.00001
        nsteps = int(dt / timestep)
        # Loop order is changed
        for i in range(nsteps):
            r_i =np.array([[p.x,p.y] for p in self.particles])

            ang_vel_i = np.array([[p.ang_vel] for p in self.particles])
            norm_i = ((r_i**2).sum(axis=1))**0.5





def visualize(simulator):
    X = [p.x for p in simulator.particles]
    Y = [p.y for p in simulator.particles]
    fig = plt.figure()
    ax = plt.subplot(111, aspect='equal')
    line, = ax.plot(X, Y, 'ro')

    # Axis limits
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # It will be run when the animation starts
    def init():
        line.set_data([], [])
        return line, # The comma is important!

    def animate(i):
        # We let the particle evolve for 0.01 time units
        simulator.evolve(0.01)
        X = [p.x for p in simulator.particles]
        Y = [p.y for p in simulator.particles]
        line.set_data(X, Y)
        return line,

    # Call the animate function each 10 ms
    anim = animation.FuncAnimation(fig,
                                    animate,
                                    init_func=init,
                                    blit=True,
                                    interval=10)
    plt.show()




def test_visualize():
    particles = [Particle(0.3, 0.5, 1),
                 Particle(0.0, -0.5, -1),
                 Particle(-0.1, -0.4, 3)]
    simulator = ParticleSimulator(particles)
    visualize(simulator)

def test_evolve():
    '''
    Unit test for particle simulator

    :return:
    '''
    particles = [Particle( 0.3, 0.5, +1),
    Particle( 0.0, -0.5, -1),
    Particle(-0.1, -0.4, +3)]
    simulator = ParticleSimulator(particles)
    simulator.evolve(0.1)
    p0, p1, p2 = particles

    def fequal(a, b, eps=1e-5):
        return abs(a - b) < eps

    assert fequal(p0.x, 0.210269)
    assert fequal(p0.y, 0.543863)
    assert fequal(p1.x, -0.099334)
    assert fequal(p1.y, -0.490034)
    assert fequal(p2.x, 0.191358)
    assert fequal(p2.y, -0.365227)

def benchmark():
    particles = [Particle(uniform(-1.0,1.0),
                          uniform(-1.0,1.0),
                          uniform(-1.0,1.0))
                                 for i in range(1000)]
    simulator = ParticleSimulator(particles)
    simulator.evolve(0.1)

def benchmark_memory():
    particles = [Particle(uniform(-1.0, 1.0),
                              uniform(-1.0, 1.0),
                              uniform(-1.0, 1.0))
                      for i in range(100000)]
    simulator = ParticleSimulator(particles)
    simulator.evolve(0.001)

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
    benchmark()
    test_evolve()
    test_visualize()