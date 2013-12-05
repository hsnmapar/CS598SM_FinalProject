%Adapted from
%ALEX DYTSO
%KALMAN FITLER PROJECT
%estimating trajectory of an object in 3-D
%http://www.mathworks.com/matlabcentral/fileexchange/36301-extended-kalman-filter-tracking-object-in-3-d
function Xh = ExtendedKalmanFilterSize(val)

Q=[ zeros(3,6);
    zeros(3),1*eye(3)];% Covarience matrix of process noise


M=.1*eye(3); % Covarience matrix of measurment noise


d=.1;% sampling time

A = [eye(3),d*eye(3);
    zeros(3),eye(3)]; % System Dynamics

start_idx = find(~any(isnan(val),2),1);
last_idx = find(~any(isnan(val),2),1,'last');
Z(:,start_idx)=val(start_idx,:);% initial observation

X(:,start_idx) = [Z(:,start_idx);0;0;0]; % "Actual" initial conditions

Xh(:,start_idx)=[Z(:,start_idx);0;0;0];%Assumed initial conditions

P(:,:,start_idx)= .1*eye(6);%inital value of covarience of estimation error

for n=start_idx:last_idx
    
    %%% Genetatubg a process and observations
    Z(:,n+1) = nextVal(val,n);
    X(1:size(Z,1),n+1) = Z(:,n+1);
    
    % PREDICTION EQUTIONS
    [Xh(:,n+1),P(:,:,n+1)]=predict(A,Xh(:,n),P(:,:,n),Q); %prediction of next state
    
    %%%%%%%%%%%%%%%%%%%
    %CORRECTION EQUTIONS
    
    H(:,:,n+1)=Jacobian(Xh(1,n+1),Xh(2,n+1),Xh(3,n+1));% subroutine computes evaluetes Jacobian matrix
    
    K(:,:,n+1)=KalmanGain(H(:,:,n+1),P(:,:,n+1),M); %subroutine computes Kalman Gain
    
    Inov=Inovation(Z(:,n+1),Xh(:,n+1)); %subroutine computes innovation
    
    Xh(:,n+1)=Xh(:,n+1)+ K(:,:,n+1)*Inov; %computes final estimate
    
    P(:,:,n+1)=(eye(6)-K(:,:,n+1)*H(:,:,n+1))*P(:,:,n+1);% %computes covarience of estimation error
    
end
if(size(Xh,2) < size(val,1))
    Xh(:,size(val,1)) = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%THIS SUBROTINE COMPUTES INNOVATION PART OF THE FILTER
    function   Inov=Inovation(Z,Xh)
        if(any(isnan(Z)))
            Inov = zeros(3,1);
            return;
        end
        hsn = [Xh(1);Xh(2);Xh(3)];
        Inov=Z-hsn;% INNOVATION
        
    end

%THIS SUBROTINE COMPUTES VALUES OF THE JACOBIAN MATRIX
    function  H=Jacobian(X,Y,Z)
        H = [eye(3),zeros(3)];
    end

%THIS SUBROTINE COMPUTES KALMAN GAIN
    function   K=KalmanGain(H,P,M)
        
        K=P*H'*(M+H*P*H')^(-1);
        
    end

%THIS SUBROTINE DOES THE PREDICTION PART OF THE KALMAN ALGORITHM
    function   [Xh,P]=predict(A,Xh,P,Q)
        Xh=A*Xh;% ESTIMATE
        
        P=A*P*A'+Q;% PRIORY ERROR COVARIENCE
    end

    function [v] = nextVal(vals, i)
        v = vals(i+1,:);
    end
end