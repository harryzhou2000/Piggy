[vol,mas] = getVM(nodes,elems);
ndof = numel(nodes);
M = [mas,mas,mas]';
MI = spdiags(1./M(:),0,ndof,ndof);


u0= [nodes(1,:) * 0; nodes(3,:) * 0; nodes(3,:) * 0];
F0 = 0 * u0;
F0(1,:) = 5e-3 * 0;
dudt0 = [nodes(1,:) * 0; nodes(3,:) * 0.1; nodes(3,:) * 0];
eta = 1e-5;
C = eye(6) * 1e2;
[iDofFix] = getDofFix(nodes);
ifDofFix = false(ndof,1);
ifDofFix(iDofFix) = true;
dofFixer = zeros(ndof,1);
dofFixer(iDofFix) = 1e10;



[f,k] = getForceKmat(u0,dudt0, F0, vol, mas, nodes,elems, eta, C,ifDofFix);
k = k + spdiags(dofFixer,0,ndof,ndof);





%% solve static
u = u0*0;
for iter = 1:100
    
    [f,k] = getForceKmat(u,dudt0*0, F0, vol, mas, nodes,elems, eta, C,ifDofFix);
    %     k = k + spdiags(dofFixer,0,ndof,ndof);
    f(iDofFix) = 0;
    du = k \ f(:);
    du(iDofFix) = 0;
    mdu = max(du(:));
    damp1 = min(0.5/(mdu+1e-10),1) * 0.5;
    
    u = u - reshape(du,3,[]) *  damp1;
    coordN = nodes + u ;
    
    if mod(iter,10) == 0
        scatter3(coordN(1,:),coordN(2,:),coordN(3,:));
        xlabel('x');ylabel('y');zlabel('z');
        axis equal;
        drawnow;
    end
    fprintf("iter %d, du %g, damp1 %g\n", iter, mdu, damp1);
    
end

%% solve dynamic explicit
t = 0;
u = u0;
dudt = dudt0;
dudt(iDofFix) = 0;
alpha = 0;



dumax = 0.5;
dvmax = 0.5/dt;
dt = 1/100;
th = 1e-3;

for iT = 1:1000
    
    f = getForceKmat(u,dudt, F0, vol, mas, nodes,elems, eta, C,ifDofFix);
    f(iDofFix) = 0;
    u1 = u + dudt * dt;
    dudt1 = dudt + reshape(MI * f(:),3,[]) * dt;
    dudt1(iDofFix) = 0;
    f1 = getForceKmat(u1,dudt1, F0, vol, mas, nodes,elems, eta, C,ifDofFix);
    f(iDofFix) = 0;
    u2    = u + (dudt + dudt1) * 0.5  *dt;
    dudt2 = dudt + reshape(MI * (f(:) + f1(:)),3,[]) * 0.5 * dt;
    dudt2(iDofFix) = 0;
    
    
    u = u2;
    dudt = dudt2;
    coordN = nodes + u ;
    
    scatter3(coordN(1,:),coordN(2,:),coordN(3,:));
    axis equal;
    drawnow;
    fprintf("===== %d\n", iT);
    
end



%% solve dynamic
t = 0;
u = u0;
dudt = dudt0;
dudt(iDofFix) = 0;
alpha = 0.0;
[f,k] = getForceKmat(u,dudt, F0, vol, mas, nodes,elems, eta, C,ifDofFix);

dt = 1/10;
dumax = 0.5;
dvmax = 0.5/dt;

th = 1e-2;

for iT = 1:1000
    uN = u;
    dudtN  = dudt;
    fold = f;
    
    incBase = 1e100;
    
    for iter = 1:100
        [f,k] = getForceKmat(uN,dudtN, F0, vol, mas, nodes,elems, eta, C,ifDofFix);
        %         k = k + spdiags(dofFixer,0,ndof,ndof);
        f(iDofFix) = 0;
        
        R1 = alpha * dudt + (1-alpha) * dudtN - (uN    -    u)/dt;
        R2 = reshape(MI * (alpha * fold(:) + (1-alpha) * f(:)),3,[])  - (dudtN - dudt)/dt;
        R1 = R1(:);
        R2 = R2(:);
        
        JCB_A = -1/dt;
        JCB_B = 1-alpha;
        JCB_C  = -1/dt - eta * (1-alpha);
        JCB_N = (1-alpha) * MI * k;
        
        NR1 = JCB_N * R1;
        DU2 = (JCB_A*JCB_C * speye(ndof,ndof) - JCB_B*JCB_N) \ (JCB_A*R2 - NR1);
        DU1 = (JCB_A*JCB_N) \ (NR1 - JCB_B * JCB_N * DU2);
        
        DU1m = max(DU1(:)) + 1e-100;
        DU2m = max(DU2(:)) + 1e-100;
        
        damp1 = min([1, dumax/DU1m, dvmax/DU2m]) * 0.9;
        uN = uN - reshape(DU1,3,[]) *  damp1;
        uN(iDofFix) = 0;
        dudtN = dudtN - reshape(DU2,3,[]) * damp1;
        dudtN(iDofFix) = 0;
        fprintf("TS %d iter %d, du %g, damp1 %g\n",iT, iter, DU1m, damp1);
        if(iter == 1)
            incBase = DU1m;
        else 
            if(DU1m < incBase * th)
               break; 
            end
        end
        
%         coordN = nodes + uN;
        
%         scatter3(coordN(1,:),coordN(2,:),coordN(3,:));
%         axis equal;
%         drawnow;
    end
    
    u = uN;
    dudt = dudtN;
    coordN = nodes + u ;
    
%     scatter3(coordN(1,:),coordN(2,:),coordN(3,:));
%     axis equal;

    pdemesh(coordN, elems);
    
    xlabel('x');ylabel('y');zlabel('z');
    view(45,45);
    drawnow;
    fprintf("=====\n");
    
end

%%


function [Bj, DiBj] = getSF()
Bj = [0.25,0.25,0.25,0.25];

DiBj = [...
    -1, 1, 0, 0
    -1, 0, 1, 0
    -1, 0, 0, 1];

end

function s = consti_F(C, e)
e6 = e([1,5,9,2,6,3]);
s6 = C * e6;
s = reshape(s6([1 4 6 4 2 5 6 5 3]),[3,3]);
end

function [iDofFix] = getDofFix(nodes)
iNodeFixB = abs(nodes(3,:) - 0) < 1e-5;
iNodes = 1:size(nodes,2);
iNodeFix = iNodes(iNodeFixB);
iDofFix = [(iNodeFix-1) * 3 + 1;(iNodeFix-1) * 3 + 2;(iNodeFix-1) * 3 + 3];
iDofFix = iDofFix(:);


end




function [vol,mas] = getVM(nodes,elems)

vol = zeros(size(elems,2),1);
mas = zeros(size(nodes,2),1);
for iElem = 1:size(elems,2)
    elem2node = elems(:,iElem);
    coords = nodes(:,elem2node);
    [Bj,DiBj] = getSF();
    Dx_i_Dxii_j = coords * DiBj';
    invJdet = det(Dx_i_Dxii_j);
    
    
    vol(iElem) =  invJdet /6;
    for i = 1:4
        mas(elem2node(i)) = mas(elem2node(i)) + vol(iElem)/4;
    end
    
end


end


function [f,K] = getForceKmat(u,dudt, f0, vol, mas, nodes,elems, damper, C, ifDofFix)


f = f0 .* mas';
f = f - dudt .* mas' * damper;
if(nargout > 1)
    Ksubs = nan(size(elems,2), 12 * 12, 2);
    Kvals = nan(size(elems,2), 12 * 12, 1);
end


for iElem = 1:size(elems,2)
    elem2node = elems(:,iElem);
    idofs = [(elem2node' - 1) * 3  + 1
        (elem2node' - 1) * 3  + 2
        (elem2node' - 1) * 3  + 3];
    idofs = idofs(:);
    ifDofFixLocal = ifDofFix(idofs);
    [Kdofi, Kdofj] = meshgrid(idofs,idofs);
    
    coords_0 = nodes(:,elem2node);
    coords_d =     u(:,elem2node);
    %     coords_1 = coords_0 + coords_d;
    
    
    [Bj,DiBj] = getSF();
    Dx0_i_Dxii_j = coords_0 * DiBj';
    %     Dx1_i_Dxii_j = coords_1 * DiBj';
    Du_i_Dxii_j = coords_d * DiBj';
    B0 = DiBj' / Dx0_i_Dxii_j;
    
    %     Dx0_i_Dx1_j = Dx0_i_Dxii_j / Dx1_i_Dxii_j;
    %     Dx1_i_Dx0_j = Dx1_i_Dxii_j / Dx0_i_Dxii_j; % F = I + Du_i_Dx0_j
    Du_i_Dx0_j = Du_i_Dxii_j / Dx0_i_Dxii_j;
    
    E_Green = (Du_i_Dx0_j' + Du_i_Dx0_j + Du_i_Dx0_j' * Du_i_Dx0_j) * 0.5;
    
    dE6_du = zeros(3,4,6);
    iss = [1,2,3,1,2,3];
    jss = [1,2,3,2,3,1];
    for iE = 1:6
        ii = iss(iE);
        jj = jss(iE);
        dE6_du(ii,:,iE) = dE6_du(ii,:,iE) + B0(:,jj)' * 0.5;
        dE6_du(jj,:,iE) = dE6_du(jj,:,iE) + B0(:,ii)' * 0.5;
        dE6_du(:,:,iE) = dE6_du(:,:,iE) + ...
            (B0(:,ii) * (coords_d * B0(:,jj))')' * 0.5;
        dE6_du(:,:,iE) = dE6_du(:,:,iE) + ...
            (B0(:,jj) * (coords_d * B0(:,ii))')' * 0.5;
    end
    dE6_duR = reshape(dE6_du, 12, 6)';
    
    E6_Green = E_Green([1,5,9,2,6,3])';
    %     T6 = C * E6_Green;
    %     W = E6_Green' * C * E6_Green;
    if(nargout > 1)
        KMat = dE6_duR' * C * dE6_duR;
        KMAX = max(abs(KMat(:)));
        KMat = (KMat + KMat')/2;
        KMat(ifDofFixLocal,:) = 0;
        KMat(:,ifDofFixLocal) = 0;
        KMat = KMat + diag(double(ifDofFixLocal)) * KMAX;
        
        Ksubs(iElem, :, 1) = Kdofi(:);
        Ksubs(iElem, :, 2) = Kdofj(:);
        Kvals(iElem, :, 1) = -KMat(:) * vol(iElem);
    end
    F = (E6_Green' * C * dE6_duR)';
    
    %     for i = 1:12
    %         f(idofs(i)) = f(idofs(i)) - F(i) * vol(iElem);
    %     end
    f(idofs) = f(idofs) - F * vol(iElem);
    
    
    %     E_Green_T = (Dx1_i_Dx0_j' * Dx1_i_Dx0_j - eye(3)) * 0.5;
    %     Sig_PK2 = consti_F(C, E_Green);
    %     Sig_PK1 = Dx0_i_Dx1_j * Sig_PK2 * Dx0_i_Dx1_j';
    
    
end

if(nargout > 1)
    K = accumarray(reshape(Ksubs,[],2),reshape(Kvals,[],1), ...
        [size(nodes,2),size(nodes,2)] * 3,...
        [],[],false);
end


end


