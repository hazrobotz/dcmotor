# Licence?
import os
import atexit
import signal
import timeit
import warnings
from sys import exc_info
import numpy as np


def save():
    np.save("saved/state.npy", xx, allow_pickle=True)
    warnings.warn("Saving plant state")


atexit.register(save)

try:
    savedstate = np.load("saved/state.npy", ndmin=2)
except:
    savedstate = None

N = 1
try:
    if "NUM_PLANTS" in os.environ.keys():
        N = int(os.environ["NUM_PLANTS"])
        warnings.warn("Using %s plants" % N)
except:
    warnings.warn("Not able to access the number of plants to generate, using %s" % N)

h = 0.05
try:
    if "SAMPLE_PERIOD" in os.environ.keys():
        _h = float(os.environ["SAMPLE_PERIOD"])
        if 0 < _h < 2:
            h = _h
        else:
            raise ValueError("Requested duration %s is not acceptable", _h)

        warnings.warn("Using %s as sample period" % h)
except:
    warnings.warn("Not able to access the suggested sample period. Using %s" % h)
clock = timeit.default_timer

# Set model parameters
R = 4.67  # ohm
L = 170e-3  # H
J = 42.6e-6  # Kg-m^2
f = 47.3e-6  # N-m/rad/sec
K = 14.7e-3  # N-m/A
Kb = 14.7e-3  # V-sec/rad

# motor_state = [theta, thetadot, i]
x = np.zeros((3, 1))
# motor_output = [velocity]
y = np.zeros((1, 1))
u = np.zeros((1, 1))

A = np.array([[0, 1, 0], [0, -f/J, K/J], [0, -K/L, -R/L]])
B = np.array([[0], [0], [1/L]])
C = np.array([[1, 0, 0],[0, 1, 0]])

xx = np.tile(x, (N, 1))
yy = np.tile(y, (N, 1))
AA = np.kron(np.eye(N), A)
BB = np.kron(np.eye(N), B)
CC = np.kron(np.eye(N), C)

# inverse Euler
I = np.identity(AA.shape[0])
Ak = np.linalg.inv(I-h*AA)
Bk = h*np.dot(np.linalg.inv(I-h*AA), BB)

t0 = clock()
t = 0
uu = np.tile(u, (N, 1))

if savedstate is not None and savedstate.shape == xx.shape and np.isreal(uu).all():
    xx = savedstate


def update(_, __):
    global yy, xx, t
    try:
        xx = np.dot(Ak, xx)+np.dot(Bk, uu)
        # xx[::3] = np.mod(xx[::3], 2*np.pi)
        yy = np.dot(CC, xx)
        t = clock() - t0
    except ValueError:
        print(exc_info(), yy, uu, "in update")


def init(data, idx=0):
    global uu, xx, t0
    data = data.split("&")
    assert 0<=idx<N, "Invalid sub-system"
    try:
        x0 = float(data[0].split("=")[1])
    except ValueError:
        x0 = 0
    uu[idx] = xx[3*idx+1] = x0
    t0 = clock()  # Time at which the controller called to initialize the plant
    return " ".join(["%.4f"%i for i in yy.flatten()[2*idx:2*idx+2]]+["%.6f"%t0])

def state(idx=0):
    assert 0<=idx<N, "Invalid sub-system"
    return " ".join(["%.4f"%i for i in yy.flatten()[2*idx:2*idx+2]]+["%.6f"%t])


def control(controldata, idx=0):
    global uu
    data = controldata.split("&")
    assert 0<=idx<N, "Invalid sub-system"

    # we will assume that all the data are present value0, and time
    try:
        u0 = float(data[0].split("=")[1])
        c_t = float(data[1].split("=")[1])
        uu[idx] = u0
        retval = " ".join(["%.4f"%i for i in yy.flatten()[2*idx:2*idx+2]]+["%.6f"%c_t])
    except (ValueError, IndexError):
        warnings.warn("%s %s %s %s" % (exc_info(), idx, "couldn't update the plant", controldata))
        retval = ""
    return retval


# method that parses the request and sends to the appropriate handler
# needed for non web triggers
def interpret(whole_data, idx):
    assert 0<=idx<N, "Invalid sub-system"
    split_data = whole_data.split("]")
    last_complete_packet = split_data[len(split_data) - 2]
 
    query_string = last_complete_packet.split('?')
 
    datahandler = query_string[0].split('/')[-1]
 
    if datahandler == "init":
        return "["+init(query_string[1], idx)+"]"
    elif datahandler == "u":
        return "["+control(query_string[1], idx)+"]"
    elif datahandler == "state":
        return "["+state(idx)+"]"
    else:
        return "["+"NOT VALID"+"]"


signal.signal(signal.SIGALRM, update)
signal.setitimer(signal.ITIMER_REAL, h, h)

if __name__ == "__main__":
    while True:
        pass
