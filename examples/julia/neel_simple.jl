using MNPDynamics
using PyPlot

# CONSTANTS
D = 20;         # particle diameter in nm
alpha = 0.1;    # damping coefficient
kAnis = 11000;  # anisotropy constant
t_length = 1000;# length of time vector
t_max = 2/25000;# maximum evaluation time in seconds
N = 20;         # maximum spherical harmonics index to be considered

t = range(0,stop=t_max,length=t_length);

# other physical parameters
D_core = D*1e-9;
V_core = pi/6 * D_core^3;
D_hydro = 35/30*D_core;
V_hydro = pi/6 * D_hydro^3;
M_S = 474000;
Temp = 293;
k_B = 1.38064852e-23;
pr1 = M_S*V_core/(k_B*Temp);
gam_gyro=1.75*10^11;
tau_N = M_S*V_core/(k_B*Temp*gam_gyro)*(1+alpha^2)/(2*alpha);
pr2 = kAnis*V_core/(k_B*Temp);
gammat = gam_gyro/(1+alpha^2);

# Anisotropy axis
n = [0;1;0];
rot = rotz(n);  # Rotation matrix that rotates n to the z axis
irot = inv(rot);# Rotation matrix that rotates the z axis to n

f = 25000;

# Magnetic field for simulation (remember to use rotation matrix if n is
# not [0;0;1])
B =  t -> rot*(0.012*[0*t; sin(2*pi*f*t); sin(2*pi*f*t)]);

@time t, y = simulation_neel(B, pr1, pr2, tau_N, alpha, t, N);

# Calculate expectation from spherical harmonics
xexptemp = real((4*pi/3)*(.5*y[:,2]-y[:,4]));
yexptemp = real(-1im*(4*pi/3)*(y[:,4]+.5*y[:,2]));
zexptemp = real((4*pi/3)*y[:,3]);

# Rotate the coordinate system back (solver only works for the z-axis as
# the easy axis)
xexp = irot[1,1]*xexptemp + irot[1,2]*yexptemp + irot[1,3]*zexptemp;
yexp = irot[2,1]*xexptemp + irot[2,2]*yexptemp + irot[2,3]*zexptemp;
zexp = irot[3,1]*xexptemp + irot[3,2]*yexptemp + irot[3,3]*zexptemp;

figure()
subplot(1,2,1)
plot(t, xexp)
plot(t, yexp)
plot(t, zexp)
#legend({'$\bar{m}_x$','$\bar{m}_y$','$\bar{m}_z$'}, 'Interpreter','latex')
subplot(1,2,2)
dt = diff(t);
dxdt = diff(xexp)./dt;
dydt = diff(yexp)./dt;
dzdt = diff(zexp)./dt;
plot(t[1:end-1], dxdt)
plot(t[1:end-1], dydt)
plot(t[1:end-1], dzdt)
#legend({'$\frac{\partial \bar{m}_x}{\partial t}$','$\frac{\partial \bar{m}_y}{\partial t}$','$\frac{\partial \bar{m}_z}{\partial t}$'}, 'Interpreter','latex')
# Note: the result has to be multiplied by M_S*V_C to obtain the magnetic
# moment
