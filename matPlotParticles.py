#%%
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import random

# --- classes ---

class Particle:
    def __init__(self):
        self.x = 5 * np.random.random_sample()
        self.y = 5 * np.random.random_sample()
        #self.vx = 5 * np.random.random_sample() - 0.5 / 5
        #self.vy = 5 * np.random.random_sample() - 0.5 / 5     
        self.vx = np.random.random_sample() / 5
        self.vy = np.random.random_sample() / 5     

    def move(self):
        if self.x < 0 or self.x >= 5:
            self.vx *= -1
        if self.y < 0 or self.y >= 5:
            self.vy *= -1
        self.x += self.vx
        self.y += self.vy

# --- functions ---

def animate(frame_number):
    global d  # need it to remove old plot

    print('frame_number:', frame_number)
    
    # move all particles
    for pi in pop:
        pi.move()

    # after for-loop    

    # remove old plot
    #d.set_data([], [])
    d.remove()
    
    # create new plot
    d, = plt.plot([particle.x for particle in pop], [particle.y for particle in pop], 'go')

# --- main ---

population = 100

pop = [Particle() for i in range(population)]

fig = plt.gcf()
# draw first plot
d,  = plt.plot([particle.x for particle in pop], [particle.y for particle in pop], 'go')
anim = animation.FuncAnimation(fig, animate, frames=200, interval=45, repeat=True)

plt.show()

#anim.save('particles.gif', fps=25)
#anim.save('particles.gif', writer='ffmpeg', fps=25)
#anim.save('particles.gif', writer='imagemagick', fps=25)
# %%
