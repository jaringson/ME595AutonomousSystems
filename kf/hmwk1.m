b = 20;
m = 100;
Ts = 0.05;

F = [-b/m 0; 1 0];
G = [1/m; 0];
H = [0 1];
J = [0];


sys = ss(F,G,H,J);
sysd = c2d(sys,Ts);
[A,B,C,D] = ssdata(sysd)