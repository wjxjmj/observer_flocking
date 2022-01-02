para=[];
para.ICL_n=100;
para.N=10;
para.stime=30;
para.Gamma=1;
para.m0=28;
para.g=9.8;
para.dim=3; %
para.r=1.2; %
para.la=1; %
para.a=10; %
para.b=1; %
para.c=50;
para.rs=2;
para.fcl_k=1;
para.enable_ICL=1;
para.ai=3.*ones(para.N,1);
para.Gamma=100;
para.rd=3;
para.hz=1/para.stime*1.8*pi;
para.Lambda=100;
para.k1=ones([para.N,1]);
para.k2=1;

%for filtered regression
para.D=zeros([para.dim,para.dim,para.N]);
para.A=zeros([para.dim,para.dim,para.N]);
for i=1:para.N
    para.D(:,:,i)=eye(para.dim);
    M = unifrnd(-3,3,[para.dim,para.dim]);
    para.A(:,:,i)=M'-M;
end

x0=zeros(para.dim,1);
theta=0.5*pi;


% initial values for states
state1=[];
state1.q = unifrnd(-5,5,[para.dim,para.N]);
state1.q(3,:)=zeros(1,para.N);
state1.p = unifrnd(-5,5,[para.dim,para.N]);
state1.p(3,:)=zeros(1,para.N);
state1.omega = unifrnd(-3,3,[para.dim,para.N]);

para.omega_0 = state1.omega;

% initial values debug;
state1.omega_f = 0*state1.omega;

% initial values for filtered regression
v_f_0 = zeros(para.dim,para.N);
% for i=1:para.N
%     v_f_0(:,i)=para.m(i)*state1.p(:,i);
% end
state1.v_f = state1.p;
state1.tau_f = zeros(para.dim,para.N);

% initial values for disturbance observer
[y0,th0]=Y_th(0,state1,para);
state1.hat_th = unifrnd(0,0,[length(th0),para.N]);

% initial values for data collection
[s0]= sliding(0,state1,para);
state1.int_Y = 0.*y0;
state1.int_s = 0.*s0;

% set memory
[xd,vd,ad,dad] = leader(0,para);
mem=[];
mem.Y=[];
mem.W=[];
mem.v = state1.tau_f + state1.v_f - state1.p;
R = Memory(mem);

% record initial values of the disturbances
para.omega_0 = state1.omega;
para.x0=mem.v;
para.s0=s0;


% simulation
sim1 = vectorField(state1,@(t,state)sys(t,state,para,R));
sim1.setFunctionAfterEachTspan(@(t,state)recording(t,state,R,para));
sim1.solve('ode113',linspace(0,para.stime,para.ICL_n+2),state1);
sim1.addSignals(@(t,s)sig(t,s,para));
[t1,result1]=sim1.result();


figure(1)
plotFormation_tk(t1,result1.q,result1.xd,-1,para)

figure(2)
plot(t1,result1.tilde_th,'-');
title('parameter convergence')

figure(3)
plot(t1,result1.err_v,'-');
title('velocity errors')

figure(4)
plot(t1,result1.tilde_d,'-');
title('disturbance estimation errors')



% figure(7)
% plot(t1,result1.dot_s,'-');hold on
% plot(t1,result1.re_dot_s,'--');hold off
% title('dot s')

% plot(t1,result1.dot_s-result1.re_dot_s,'-');

% figure(8)
% compareGradient(result1.s,result1.dot_s,t1);
% 
% figure(9)
% plot(t1,result1.s,'-');hold on
% plot(t1,result1.re_s,'--');hold off
% title('s')
% 
% figure(10)
% compareGradient(result1.int_s,result1.s,t1);
% 
% figure(11)
% compareGradient(result1.int_v,result1.v,t1);
% 
% figure(11)
% compareGradient(result1.int_dis,result1.dis,t1);

% save([date,'.mat'])

function signals = sig(t,state,para)
[xd,vd,~,~] = leader(t,para);
% [~,u]=sys(t,state,para);
[Y_i,th]=Y_th(t,state,para);
% v_rebuilt_t = zeros(para.dim,para.N);
d_t = zeros(para.dim,para.N);
% re_d_t = zeros(para.dim,para.N);
hat_d_t = zeros(para.dim,para.N);
tilde_th = zeros([length(th),para.N]);
% [s,dxi] = sliding(t,state,para);
% dot_s = zeros([para.dim,para.N]);
% re_s_t = zeros([para.dim,para.N]);
% v_t = zeros([para.dim,para.N]);
% int_v_t = zeros([para.dim,para.N]);
% dot_s1 = zeros([para.dim,para.N]);
% dis_t = zeros([para.dim,para.N]);
% int_dis_t = zeros([para.dim,para.N]);
omega_th2 = zeros([para.dim,para.dim]);
for i=1:para.N
    Di = para.D(:,:,i);
%     Ai = para.A(:,:,i);
    ai=para.ai(i);
%     v_rebuilt_t(:,i)=state.v_f(:,i)-Di * state.omega_f(:,i)+state.tau_f(:,i);
%     
    d_t(:,i) = Di * state.omega(:,i);
%     re_d_t(:,i) = ai*Di*state.omega_f(:,i)+Di*Ai*state.omega_f(:,i)+exp(-ai*t)*Di*para.omega_0(:,i);
    hat_d_t(:,i) = ai*Di*state.omega_f(:,i)+Y_i(:,:,i)*state.hat_th(:,i);
    tilde_th(:,i) = th(:,i) - state.hat_th(:,i);
    omega_th2(:,i) = state.omega(:,i)'*state.omega(:,i);
%     
end

% signals.rebuil_p=v_rebuilt_t;

signals.d = d_t;
signals.hat_d = hat_d_t;
signals.tilde_d = d_t - hat_d_t;

signals.err_v = state.p - repmat(vd,[1,para.N]);

signals.tilde_th = tilde_th;

signals.omega_th2 = omega_th2;

signals.xd=xd;
signals.vd=vd;
end

function new_state = recording(t,state,mem,para)
new_state = state;
new_state.int_Y = 0.*state.int_Y;
new_state.int_s = 0.*state.int_s;
index = mem.n+1;
v_t = state.tau_f + state.v_f - state.p;
v_pre = mem.data.v;
% s_pre = mem.data.s;
mem.data.W{index}=v_t - v_pre;
mem.data.Y{index}=state.int_Y;
mem.n=mem.n+1;
fprintf('%d\n',mem.n+1);
mem.data.v = state.tau_f + state.v_f - state.p;
% mem.data.s = s_t;
end

function [grad,u] = sys(t,state,para,mem)
x=state.q;
v=state.p;

v_f=state.v_f;
tau_f=state.tau_f;

omega = state.omega;

hat_th = state.hat_th;

% debug
omega_f = state.omega_f;

dot_v_f=zeros(size(v_f));
dot_tau_f=zeros(size(tau_f));
dot_hat_th=zeros(size(hat_th));
dot_omega = zeros(size(omega));

% debug
dot_omega_f = zeros(size(omega_f));
%

[xd,vd,ad,dad] = leader(t,para);


u=zeros(size(state.p));
u_dot_q=zeros(size(state.p));
dot_hat_th = zeros(size(hat_th));
[Y_i,th_i]=Y_th(t,state,para);
[s,dxi] = sliding(t,state,para);
for i=1:para.N
    
    Di = para.D(:,:,i);
    Ai = para.A(:,:,i);
    
    xi=x(:,i);
    vi=v(:,i);
    
    xi_i = xi - xd - vd;
    dxi_i= vi - vd - ad;
    
    for j=1:para.N
        xj = x(:,j);
        vj = v(:,j);
        
        xi_i = xi_i + phi(xi-xj,50,para.r,para.la);
        dxi_i =dxi_i+ dPhi(xi-xj,vi-vj,50,para.r,para.la);
    end
    
    % sliding variable
    si=s(:,i);
    
    %observer
    hat_d_i = para.ai(i)*Di*omega_f(:,i)+Y_i(:,:,i)*hat_th(:,i);
    
    %controller

    u(:,i) = -para.k1(i)*si-para.k2*(vi-vd)-dxi(:,i)+hat_d_i;
    
    % adaptive law for uncertainties
    dot_hat_th(:,i) = para.Lambda*Y_i(:,:,i)'*si;
    
    % filtered regressor
    dot_v_f(:,i) = -para.ai(i) * v_f(:,i) + para.ai(i)*vi;
    dot_tau_f(:,i) = -para.ai(i) * tau_f(:,i)+u(:,i);
    
    % debug
    dot_omega_f(:,i)=-para.ai(i) * omega_f(:,i) + omega(:,i);
    %
    
    % adaptive law
    dot_hat_th(:,i) = -para.Gamma*Y_i(:,:,i)'*(si);
    %     dot_bar_omega_f(:,i)=-para.ai * bar_omega_f(:,i)+hat_omega_i;
    
    % filtered concurrent learning adaptive law
    if para.enable_ICL==1
        LAM = [];
        u_icl=[];
        for k=1:mem.n
            yk=mem.data.Y{k};
            wk=mem.data.W{k};
            yk_i=yk(:,:,i);
            wk_i=wk(:,i);
            
            if k==1
                LAM=yk_i'*yk_i;
                u_icl=yk_i'*(wk_i-yk_i*hat_th(:,i));
%                 tilde_th_i = th_i(:,i)-hat_th(:,i);
%                 a1=yk_i*th_i(:,i);
%                 a2=wk_i;
%                 a3=wk_i-yk_i*hat_th(:,i);
%                 a4=yk_i*tilde_th_i;
%                 if t>20
%                     t;
%                 end
            else
                LAM=LAM+yk_i'*yk_i;
                u_icl=u_icl+yk_i'*(wk_i-yk_i*hat_th(:,i));
%                 tilde_th_i = th_i(:,i)-hat_th(:,i);
%                 a1=yk_i*th_i(:,i);
%                 a2=wk_i;
%                 a3=wk_i-yk_i*hat_th(:,i);
%                 a4=yk_i*tilde_th_i;
%                 if t>20
%                     t;
%                 end
            end
        end
        if mem.n>0
            dot_hat_th(:,i)=dot_hat_th(:,i)+para.Gamma*pinv(LAM)*u_icl;
        end
    end
    
%     % check
%     if t>0.05
%         Di = para.D(:,:,i);
%         Ai = para.A(:,:,i);
%         ai=para.ai(i);
%         tilde_th_i = th_i(:,i) - hat_th(:,i);
%         e1 = dxi(:,i)+u(:,i)-Di*omega(:,i);
%         e2 = -para.k1(i)*si-para.k2*(v(:,i)-vd)-Y_i(:,:,i)*tilde_th_i;
%         e3 = -para.k1(i)*si-para.k2*(vi-vd)+hat_d_i-Di*omega(:,i);
%         
%         
%         b1 = si-para.s0(:,i);
%         b2 =-para.k1(i)*state.int_s(:,i)...
%             -para.k2*(xi-xd-(para.x0(:,i)))...
%             -state.int_Y(:,:,i)*tilde_th_i;
%         t;
%     end
    
    % EL dynamic
    u_dot_q(:,i) = (u(:,i)-Di*omega(:,i));
    
    % disturbance
    dot_omega(:,i) = Ai * omega(:,i);
end

grad.q=v;
grad.p=u_dot_q;
grad.v_f=dot_v_f;
grad.tau_f=dot_tau_f;
grad.hat_th=dot_hat_th;
% grad.bar_omega_f=dot_bar_omega_f;
grad.omega = dot_omega;
% debug
grad.omega_f = dot_omega_f;
%
grad.int_Y = Y_i;
grad.int_s = s;
end

function [s,dxi] = sliding(t,state,para) 
s=zeros(para.dim,para.N);
dxi=zeros(para.dim,para.N);
[xd,vd,ad,~] = leader(t,para);
for i=1:para.N 
    xi=state.q(:,i);
    vi=state.p(:,i);
    
    xi_i = xi - xd - vd;
    dxi_i= vi - vd - ad;
    
    for j=1:para.N
        xj = state.q(:,j);
        vj = state.p(:,j);

        xi_i = xi_i + phi(xi-xj,50,para.r,para.la);
        dxi_i =dxi_i+ dPhi(xi-xj,vi-vj,50,para.r,para.la);
    end
    s(:,i)=xi_i+vi;
    dxi(:,i)=dxi_i;
end
end

function [Ys,ths]=Y_th(t,state,para)
for i=1:para.N
    D_i = para.D(:,:,i);
    A_i = para.A(:,:,i);
    omega_f_i = D_i\(state.v_f(:,i)+state.tau_f(:,i)-state.p(:,i));
    Y_1 = kron(omega_f_i',D_i);
    th_1 = vec(A_i);
    Y_2 = exp(-para.ai(i)*t)*D_i;
    th_2 = para.omega_0(:,i);
    Y_i = [Y_1,Y_2];
    th_i = [th_1;th_2];
    if i==1
        Ys = zeros([size(Y_i),para.N]);
        ths= zeros(length(th_i),para.N);
    end
    Ys(:,:,i) = Y_i;
    ths(:,i) = th_i;
end
end

function [y,dy]=leaderForce(x,v,xd,vd,rs,c)

% e=0.01;
% [d,n]=size(x);
% y=zeros(size(x));
% dy=zeros(size(x));
% for i=1:n
%     xid=x(:,i)-xd;
%     vid=v(:,i)-vd;
%     dis2=xid'*xid;
%     dis=dis2^0.5;
%     dir   = xid;
%     d_dir = vid;
%     y(:,i)=c*phid(dis/rs)*dir;
%     dy(:,i)=c*(1/rs*dPhid(dis)*xid'*vid/dis+phid(dis/rs))*d_dir;
% end

% e=0.01;
% [d,n]=size(x);
% y=zeros(size(x));
% dy=zeros(size(x));
% for i=1:n
%     xid=x(:,i)-xd;
%     vid=v(:,i)-vd;
%     dis2=xid'*xid;
%     dis=dis2^0.5;
%     dir   = xid;
%     d_dir = vid;
%     y(:,i)=c*phid(dis/rs)*dir;
%     dy(:,i)=c*(1/rs*dPhid(dis)*xid'*vid/dis+phid(dis/rs))*d_dir;
% end


% e=0.01;
% [d,n]=size(x);
% y=zeros(size(x));
% dy=zeros(size(x));
% A=diag([1,1,1]);
% for i=1:n
%     xid=x(:,i)-xd;
%     vid=v(:,i)-vd;
%     dis2=xid'*A*xid;
%     dis=dis2^0.5;
%     dir   = xid/(dis+e);
%     d_dir = (-1/(dis*(e+dis)^2).*xid*xid'+1/(e+dis).*eye(size(xid,1)))*vid;
%     y(:,i)=phid(dis/rs)*dir;
%     dy(:,i)=1/rs*dPhid(dis)*xid'*A*vid/dis+phid(dis/rs)*d_dir;
% end
%
y  = x-repmat(xd,[1,size(x,2)]);
dy = v-repmat(vd,[1,size(v,2)]);
end

function [y,dy]=attr(x,v,xd,vd,rs)
e=0.01;
[d,n]=size(x);
y=zeros(size(x));
dy=zeros(size(x));

for i=1:n
    xid=x(:,i)-xd;
    vid=v(:,i)-vd;
    dis=norm(xid);
    dir   = xid/(dis+e);
    d_dir = (-1/(dis*(e+dis)^2).*xid*xid'+1/(e+dis).*eye(size(xid,1)))*vid;
    y(:,i)=phid(dis/rs)*dir;
    dy(:,i)=1/rs*dPhid(dis)*xid'*vid/dis+phid(dis/rs)*d_dir;
end
end

function [y,dy]=agentForce2(x,v,r,la,a,b)
e=0.01;
n = size(x,2);
y = zeros(size(v));
dy= zeros(size(v));
for i=1:n
    xi=x(:,i);
    vi=v(:,i);
    for j=1:n
        if i==j
            continue
        else
            xj=x(:,j);
            vj=v(:,j);
            xij=xi-xj;
            vij=vi-vj;
            dis = norm(xi-xj);
            if dis<r
                g=b;
                if dis>la
                    g=a;
                end
                aij   = rho_h(dis^2,r^2,0.8);
                d_aij = d_rho_h(dis^2,r^2,0.8)*2*xij'*vij;
                arij  = g*((dis^2-la^2)*pi);
                d_arij= g*pi*2*xij'*vij;
                dir   = xij/(dis+e);
                d_dir = (-1/(dis*(e+dis)^2).*xij*xij'+1/(e+dis).*eye(size(xij,1)))*vij;
                y(:,i) = y(:,i);
                dy(:,i)=dy(:,i)+d_aij*arij*dir+aij*d_arij*dir+aij*arij*d_dir;
            end
        end
    end
end
end

function [y,dy]=agentForce(x,v,r,la,a,b)
e=0.01;
n = size(x,2);
y = zeros(size(v));
dy= zeros(size(v));
for i=1:n
    xi=x(:,i);
    vi=v(:,i);
    for j=1:n
        if i==j
            continue
        else
            xj=x(:,j);
            vj=v(:,j);
            xij=xi-xj;
            vij=vi-vj;
            dis = norm(xi-xj);
            if dis<r
                g=b;
                if dis>la
                    g=a;
                end
                aij   = rho_h(dis^2,r^2,0.8);
                d_aij = d_rho_h(dis^2,r^2,0.8)*2*xij'*vij;
                arij  = g*((dis^2-la^2)*pi);
                d_arij= g*pi*2*xij'*vij;
                dir   = xij/(dis+e);
                d_dir = (-1/(dis*(e+dis)^2).*xij*xij'+1/(e+dis).*eye(size(xij,1)))*vij;
                y(:,i) = y(:,i)+phi(dis,la,r,a,b)*dir;
                dy(:,i)=dy(:,i)+dPhi(dis,la,r,a,b)*xij'*vij/dis*dir+phi(dis,la,r,a,b)*d_dir;
            end
        end
    end
end
end

function y=smoothUp(x)
if x>1
    y=1;
elseif x<0
    y=0;
else
    y=x-sin(2*pi*x)/(2*pi);
end
end
function y=smoothDown(x)
if x>1
    y=0;
elseif x<0
    y=1;
else
    y=1-x+sin(2*pi*x)/(2*pi);
end
end
function y=smoothUpDown(x,h1,h2)
if x<h1
    y=smoothUp(x/h1);
elseif x<h2
    y=1;
else
    y=smoothDown((x-h2)/(1-h2));
end
end

function y=dSmoothUp(x)
if x>1
    y=0;
elseif x<0
    y=0;
else
    y=2*sin(pi*x)^2;
end
end
function y=dSmoothDown(x)
if x>1
    y=0;
elseif x<0
    y=0;
else
    y=-1+cos(2*pi*x);
end
end
function y=dSmoothUpDown(x,h1,h2)
if x<h1
    y=1/h1*dSmoothUp(x/h1);
elseif x<h2
    y=0;
else
    y=1/(1-h2)*dSmoothDown((x-h2)/(1-h2));
end
end

function y=rho_h(z,r,h)
if z<h*r && z>=0
    y=1;
elseif z<=r && z>=h*r
    y=(2*pi*(-r+z)+(-1+h)*r*sin(2*pi*(-h*r+z)/(r-h*r)))/(2*(h-1)*r*pi);
else
    y=0;
end
end

function y=d_rho_h(z,r,h)
if z<h*r && z>=0
    y=0;
elseif z<=r && z>=h*r
    y=2*sin(pi*(-h*r+z)/(r-r*h))^2/((-1+h)*r);
else
    y=0;
end
end


function plotFormation(t,data,methodName,para)

dim=length(data.xd(end,:));
plot3(data.q(:,1:dim:end),data.q(:,2:dim:end),data.q(:,3:dim:end));
plot3(data.xd(:,1:dim:end),data.xd(:,2:dim:end),data.xd(:,3:dim:end),'r--');
plot3(data.q(end,1:dim:end),data.q(end,2:dim:end),data.q(end,3:dim:end),'o');hold on
plot3(data.xd(end,1:dim:end),data.xd(end,2:dim:end),data.xd(end,3:dim:end),'rp');
n=numel(data.q(end,:))/dim;
q=reshape(data.q(end,:),[dim,n]);
for i=1:n
    qi=q(:,i);
    for j=1:n
        qj=q(:,j);
        if i==j
            continue
        else
            dis=norm(qi-qj);
            if dis<para.la*0.95
                line([qi(1);qj(1)],[qi(2);qj(2)],[qi(3);qj(3)],'color','r')
            elseif dis<para.r
                line([qi(1);qj(1)],[qi(2);qj(2)],[qi(3);qj(3)],'color','b')
            end
        end
    end
end
hold off
title(['formation (',methodName,')'])
grid on
axis equal
end

function [xd,vd,ad,dad]=leader(t,para)
rd = para.rd;
hz = para.hz;
xd=[rd*cos(t*hz);rd*sin(t*hz);0];
vd=[-rd*hz*sin(t*hz);rd*hz*cos(t*hz);0];
ad=[-rd*hz^2*cos(t*hz);-rd*hz^2*sin(t*hz);0];
dad=[rd*hz^3*sin(t*hz);-rd*hz^3*cos(t*hz);0];
end

function plotFormation_tk(t,q,xd,ts,para)

dim=para.dim;

plot3(xd(:,1:dim:end),xd(:,2:dim:end),xd(:,3:dim:end),'r--','linewidth',1);hold on
plot3(q(:,1:dim:end),q(:,2:dim:end),q(:,3:dim:end),'c-');
plot3(q(1,1:dim:end),q(1,2:dim:end),q(1,3:dim:end),'kx');
hold on
if ts~=-1
    for k=1:length(ts)
        ts_i = ts(k);
        t_delta = t - ts_i;
        t_delta = abs(t_delta);
        [~,index]=min(t_delta);
        t_i = index;
        plotFormationAtTime(q(t_i,:),xd(t_i,:),para)
    end
end

plotFormationAtTime(q(end,:),xd(end,:),para)

hold off
% title(['formation (',methodName,')'])
grid on
axis equal
end


function plotFormationAtTime(q,xd,para)
dim=para.dim;
n = para.N;

plot3(q(1:dim:end),q(2:dim:end),q(3:dim:end),'o','Color',[1,1,1],'MarkerFaceColor',[0.2,0.6,1]);
plot3(xd(1:dim:end),xd(2:dim:end),xd(3:dim:end),'p','Color','r','MarkerFaceColor','r');
q=reshape(q,[dim,n]);
for i=1:para.N
    xi = q(:,i);
    for j=1:para.N
        xj=q(:,j);
        d = norm(xi-xj);
        if abs(d-para.la)<0.1
            line([xi(1),xj(1)],[xi(2),xj(2)],[xi(3),xj(3)],'Color',[0,0,0]);
        end
    end
end

end

function y = phi(x,k,r,l)
dis2 = x'*x;
y = k * rho(dis2/r^2)*(dis2-l^2)*x;
end

function y = dPhi(x,v,k,r,l)
dis2 = x'*x;
y =  k * dRho(dis2/r^2)*2*x'*v/r^2*(dis2-l^2)*x+...
    k * rho(dis2/r^2)*(2*x'*v)*x+...
    k * rho(dis2/r^2)*(dis2-l^2)*v;
end

function y = rho(x)
if x<1
    y=1-x+sin(2*pi*x)/2/pi;
else
    y=0;
end
end

function y = dRho(x)
if x<1
    y=-1 + cos(2*pi*x);
else
    y=0;
end
end

function y = ddRho(x)
if x<1
    y= -2*pi * sin(2*pi*x);
else
    y=0;
end
end