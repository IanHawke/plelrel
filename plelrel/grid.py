import numpy

class grid(object):
    def __init__(self, interval, Npoints, Ngz):
        self.interval = interval
        self.Npoints = Npoints
        self.Ngz = Ngz
        self.dx = (self.interval[1] - self.interval[0]) / self.Npoints
    def coordinates(self):
        x_start = self.interval[0] + (0.5 - self.Ngz) * self.dx
        x_end   = self.interval[1] + (self.Ngz - 0.5) * self.dx
        return numpy.linspace(x_start, x_end, self.Npoints + 2 * self.Ngz)
    def interior_coordinates(self):
        x_start = self.interval[0] + 0.5 * self.dx
        x_end   = self.interval[1] - 0.5 * self.dx
        return numpy.linspace(x_start, x_end, self.Npoints)
