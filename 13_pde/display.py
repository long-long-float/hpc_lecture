import numpy
from matplotlib import pyplot, cm
import json
import sys

fname = sys.argv[1]

with open(fname) as f:
    data = json.loads(f.read())
    X, Y = numpy.meshgrid(data['x'], data['y'])
    p, u, v = numpy.array(data['p']), numpy.array(data['u']), numpy.array(data['v'])

    fig = pyplot.figure(figsize=(11,7), dpi=100)
    # plotting the pressure field as a contour
    pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    pyplot.colorbar()
    # plotting the pressure field outlines
    pyplot.contour(X, Y, p, cmap=cm.viridis)
    #import pdb; pdb.set_trace()
    # plotting velocity field
    pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    pyplot.xlabel('X')
    pyplot.ylabel('Y')

    pyplot.savefig(fname + ".png")

