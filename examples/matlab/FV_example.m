clear
close all
tic

addpath('../../matlab')
load('../../matlab/FV_meshes/mesh_4.mat')
clear tr

tr.a_ijs = a_ijs;
tr.areas = areas;
tr.areasi = areasi;
tr.areasidil = areasidil;
tr.C = C;
tr.centers = centers;
tr.ds = ds;
tr.depth_level = depth_level;
tr.e_is = e_is;
tr.edge_dist = edge_dist;
tr.edgeAttachments = edgeAttachments;
tr.edges = edges;
tr.flow_signs = flow_signs;
tr.fMat = fMat;
tr.iis = iis;
tr.is = is;
tr.js = js;
tr.mids = mids;
tr.neighs = neighs;
tr.tr2edge = tr2edge;
tr.valcs = valcs;
tr.vMat = vMat;


params.n = [0;0;1];
f = 25000;
B =@(t)0.012*[cos(.34*2*pi*f*t)-.05;sin(.7*2*pi*f*t)+.1;sin(2*pi*f*t)];

t = linspace(0,4/f, 1000);

[t, yexp, y] = simulation_FV(B, t, tr, params);

total = zeros(size(t));
for i=1:length(t)
    total(i) = y(i,:)*areas';
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
