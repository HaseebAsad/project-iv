import pyglet
import pyglet.gl as gl 
import math

#Trying the Mattia conti approach.

class particle:

    def __init__(self, position, mass, velocity, index, color = (0, 0, 255)):
        self.position = position #initial position
        self.mass = mass #mass, will just set to 1
        self.velocity = velocity #initial velocity?
        self.speed = math.sqrt(self.velocity[0]**2+self.velocity[1]**2)

        self.index = index
        self.isscatter = False

        self.pointsdraw = pyglet.graphics.vertex_list(1, ('v2f/stream', self.position), ('c3B', color) ) # colour is blue Does not work because v1

    # Moves the particle given its velocity
    def move(self):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]
        self.pointsdraw.vertices = self.position

    # Detect collisions
    def intersection(self, particles, nparticles):
        for i in range(nparticles): #Check intersections for all particles
            if i != self.index:
                x = self.position[0] - particles[i].position[0]
                y = self.position[1] - particles[i].position[1]
                if x ** 2 + y**2 < 1: #if in radius of 1
                    return True #collision
        return False #No collision
    
    def scatter(self, particles, nparticles): #Elastic collision
        where = self.intersection(particles, nparticles)
        print(where[0])
        if where[0] and particles[where[1]].isscatter:

            # Avoid multiple scattering at the same instant
            particles[where[1]].isscatter = False
            self.isscatter = False

            totalmass = self.mass + particles[where[1]].mass
            massdiff = self.mass - particles[where[1]].mass
            
            #Store the velocity variable temporarily
            tempvelocity = [self.velocity[0], self.velocity[1]]
            self.velocity[0] = (massdiff*self.velocity[0] + 2*particles[where[1]].mass*particles[where[1]].velocity[0])/totalmass
            self.velocity[1] = (massdiff*self.velocity[1] + 2*particles[where[1]].mass*particles[where[1]].velocity[1])/totalmass
            particles[where[1]].velocity[0] = (-massdiff * tempvelocity[0] + 2*self.mass*tempvelocity[0])/totalmass
            particles[where[1]].velocity[1] = (-massdiff * tempvelocity[1] + 2*self.mass*tempvelocity[1])/totalmass



# Resize the window (problems with pyglet)
def resize(width, height, zoom, x, y):
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.glOrtho(-width, width, -height, height, -1, 1)
    gl.glViewport(0, 0, width, height)
    gl.glOrtho(-zoom, zoom, -zoom, zoom, -1, 1)
    gl.glTranslated(-x, -y, 0)

class mywindow(pyglet.window.Window):

    def __init__(self, width, height, name):
        super().__init__(width, height, name, resizable=True)

        gl.glClearColor(1, 1, 1, 1)

        #gl.glpointsize(5) #Does not work because this is not supported in current version 1.5.27?
        # Window things
        self.width = width
        self.height = height
        self.name = name
        self.zoom = 1
        self.x = 0
        self.y = 0
        self.time = 0

        self.key = None

        self.Nparticles = 2
        self.rangeparticles = range(self.Nparticles)
        self.mainparticles = []
        self.mainparticles += [particle([-self.width/2,0], 30, [1,0], 0)]
        self.mainparticles += [particle([-self.width/2,0], 10, [-1,0], 1, [255, 0, 0])] #red


    def on_draw(self, dt = 0.002): #timestep
        self.clear()
        for i in self.rangeparticles:
            self.mainparticles[i].pointsdraw.draw(pyglet.gl.GL_POINTS)

        for i in self.rangeparticles:
            self.mainparticles[i].scatter(self.mainparticles, self.Nparticles)

        for i in self.rangeparticles:
            self.mainparticles[i].isscatter = True
            self.mainparticle.move()

    def on_resize(self, width, height):
        gl.glMatrixMode(gl.GL_MODELVIEW) 
        gl.glLoadIdentity() 
        gl.glOrtho(-width, width, -height, height, -1, 1)
        gl.glViewport(0, 0, width, height)
        gl.glOrtho(-self.zoom, self.zoom, -self.zoom, self.zoom, -1, 1)

mywindow(3, 3, 'simulation')
pyglet.app.run()

# Errors occur because Pyglet version is too new. Need pyglet version 1 when we have version 2.