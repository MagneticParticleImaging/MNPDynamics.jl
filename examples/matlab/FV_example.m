clear
close all
tic

addpath('../../matlab')
load('../../matlab/FV_meshes/mesh_4.mat')

params.kAnis = 11000;
%params.n = [0;0;1];
f = 25000;
%params.n = @(t) [cos(2*pi*f*t);sin(2*pi*f*t);0];
B =@(t)0.012*[cos(.34*2*pi*f*t)-.05;sin(.7*2*pi*f*t)+.1;sin(2*pi*f*t)];
params.p1 = 0;
params.p3 = 0;
%B = @(t)0.012*[0;0;sin(2*pi*f*t)];

t = linspace(0,4/f, 1000);

[t, yexp, y] = simulation_FV(B, t, tr, params);

total = zeros(size(t));
for i=1:length(t)
    total(i) = y(i,:)*tr.areas';
end
plot(t, total)
figure
plot(t, yexp)
pause(1)
figure
plot(t(1:end-1), diff(yexp(:,3)))

% 
% figure
% for i=1:length(t)
%     trisurf(fMat, vMat(:,1), vMat(:,2), vMat(:,3), y(i,:), 'EdgeColor', 'none')
%     title(num2str(i/length(t)))
%     caxis([min(min(y)), max(max(y))]);
%     colorbar()
%     drawnow()
% end
% 
