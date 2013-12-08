%-------------------------------------------------------------------------%
% University of Illinois
% Joseph DeGol - Fall 2013
%-------------------------------------------------------------------------%
% 
% P5_JMD.m
% Class that provides the methods to complete project 5. This includes
% methods for Image Stitching and homography estimation
%
%-------------------------------------------------------------------------%

function myP5 = P5_JMD()

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%   Interface Decleration   %%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %---------------------------------------------------------%
    %------------------------ Methods ------------------------%
    %---------------------------------------------------------%
    
    %----- Prime -----%
    myP5.Prime = @Prime;
    %--- End Prime ---%
    
    %----- Stitch -----%
    myP5.User_Stitch       = @UserStitch;
    myP5.User_Paste        = @UserPaste;
    myP5.Auto_Stitch       = @AutoStitch;
    myP5.Auto_Stitch_Multi = @AutoStitchMulti;
    myP5.Pose_Estimate     = @PoseEstimate;
    %--- End Stitch ---%
    
    %----- Display -----%
    myP5.Display_Matches = @DisplayMatches;
    myP5.Display_Inliers = @DisplayInliers;
    myP5.Display_Pose    = @DisplayPose;
    %--- End Display ---%
    
    %---------------------------------------------------------%
    %------------------------ Methods ------------------------%
    %---------------------------------------------------------%
    
    
    
    %---------------------------------------------------------%
    %------------------------ Helpers ------------------------%
    %---------------------------------------------------------%
    
    myP5.Compute_Homography_RANSAC = @ComputeHomographyRANSAC;
    myP5.Compute_Homography        = @ComputerHomography;
    myP5.Compute_Pose              = @ComputePose;
    myP5.Interest_Points           = @InterestPoints;
    myP5.Stitch_Planar             = @StitchPlanar;
    myP5.Stitch_Planar_Multi       = @StitchPlanarMulti;
    myP5.Billboard_Paste           = @BillboardPaste;
    
    %---------------------------------------------------------%
    %---------------------- End Helpers ----------------------%
    %---------------------------------------------------------%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%   Interface Functions   %%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    

    
    %% Prime
    %---------------------------------------------------------------------%
    %------------------------------ Prime --------------------------------%
    %---------------------------------------------------------------------%
    
    %----- Prime -----%
    % creates the P5 Object
    %
    % input:
    %    Images - cell of image name strings
    % output:
    %     P5Obj - primed P5Obj
    function P5Obj = Prime(Images)
        
        if nargin < 1 || isempty(Images), 
            fprintf('o    P5_JMD: Prime Skipped. Not enough Image files given.\n');
        else
            %read in images
            Ims = cell(length(Images),1);
            for i=1:length(Images),
                Ims{i} = imread(Images{i});
            end
            
            %return
            P5Obj.Images = Ims;
        end
    end
    %--- End Prime ---%

    %---------------------------------------------------------------------%
    %---------------------------- End Prime ------------------------------%
    %---------------------------------------------------------------------%
    
    
    
    
    %% Stitch
    %---------------------------------------------------------------------%
    %------------------------------ Stitch -------------------------------%
    %---------------------------------------------------------------------%
    
    %----- User Stitch -----%
    % stitches two images together from user defined correspondences
    %
    % input:
    %    P5Obj - primed P5Obj with Images field
    %    N - number of user correspondences to use
    % output:
    %    P5Obj - new field user_stitch
    function P5Obj = UserStitch(P5Obj,N)
        
        %Get Correspondences
        Corr = zeros(N,4);
        imshow([P5Obj.Images{1} P5Obj.Images{2}]); hold on;
        [~, c, ~] = size(P5Obj.Images{1});
        for i=1:N,
            title('mark point');
            [x1, y1] = ginput(1);
            plot(x1,y1,'go'); drawnow;
            title('mark correspondence');
            [x2, y2] = ginput(1);
            Corr(i,:) = [x1 y1 x2-c y2];
            plot(x2,y2,'r*');
            line([x1 x2],[y1 y2],'Color','b');
        end
        
        %Get H
        H = ComputeHomography(Corr(:,3:4),Corr(:,1:2));

        %Stitch
        User_Stitched = StitchPlanar(P5Obj.Images,H);

        %return
        P5Obj.User_Stitch = User_Stitched;
    end
    %--- End User Stitch ---%
    
    %----- User Paste -----%
    % pastes an image into a scene with a defined rectangular region
    %
    % input:
    %    P5Obj - primed P5Obj with Images field
    %    N - number of times we want to paste
    % output:
    %    P5Obj - new field user_stitch
    function P5Obj = UserPaste(P5Obj,N)
        if nargin < 2, N = 1; end
        
        for i=1:N,
            
            if i==1, Images = { P5Obj.Images{1} P5Obj.Images{2} };
            else Images = { P5Obj.User_Stitch P5Obj.Images{2} }; end
            
            %Get Correspondences
            Corr = zeros(4,4);
            imshow(Images{1}); hold on;
            [r,c,~] = size(Images{1});

            %top left
            title('mark top left');
            [x1, y1] = ginput(1);
            plot(x1,y1,'go'); drawnow;
            Corr(1,:) = [x1 y1 0 0];

            %top right
            title('mark top right');
            [x1, y1] = ginput(1);
            plot(x1,y1,'go'); drawnow;
            Corr(2,:) = [x1 y1 c 0];

            %bottom left
            title('mark bottom left');
            [x1, y1] = ginput(1);
            plot(x1,y1,'go'); drawnow;
            Corr(3,:) = [x1 y1 0 r];

            %bottom right
            title('mark bottom right');
            [x1, y1] = ginput(1);
            plot(x1,y1,'go'); drawnow;
            Corr(4,:) = [x1 y1 c r];

            %Get H
            H = ComputeHomography(Corr(:,3:4),Corr(:,1:2));

            %Stitch
            User_Stitched = BillboardPaste(Images,H);

            %return
            P5Obj.User_Stitch = User_Stitched;
        end
    end
    %--- End User Stitch ---%

    %----- Auto Stitch -----%
    % stitches two images together from sift features and RANSAC homography
    %
    % input:
    %    P5Obj - primed P5Obj with Images field
    % output:
    %    P5Obj - new field auto_stitch
    function P5Obj = AutoStitch(P5Obj)
        
        %Get Correspondences
        Dist = 0.7;
        Corr = InterestPoints(P5Obj.Images, Dist);
        
        %Get H
        Max_Iterations = 2000;
        Inlier_Threshold = 1;
        [H, Inliers_Indices] = ComputeHomographyRANSAC( Corr, Max_Iterations, Inlier_Threshold);
        DisplayInliers(P5Obj.Images,Corr,Inliers_Indices)
        
        %Stitch
        Auto_Stitched = StitchPlanar(P5Obj.Images,H);

        %return
        P5Obj.Auto_Stitch = Auto_Stitched;
        
    end
    %--- End Auto Stitch ---%
    
    %----- Auto Stitch Multi -----%
    % stitches multiple images together from sift features and RANSAC
    % homographies
    %
    % input:
    %    P5Obj - primed P5Obj with Images field
    % output:
    %    P5Obj - new field auto_stitch
    function P5Obj = AutoStitchMulti(P5Obj)
        
        %Settings
        Dist = 0.7;
        Max_Iterations = 2000;
        Inlier_Threshold = 1;
        H = cell(length(P5Obj.Images)-1,1);
        
        %For each pair of images
        for i=1:length(P5Obj.Images)-1,
            
            %Get Correspondences
            Images = {P5Obj.Images{i} P5Obj.Images{i+1}};
            Corr = InterestPoints(Images, Dist);
            
            %Get H
            [H_, ~] = ComputeHomographyRANSAC( Corr, Max_Iterations, Inlier_Threshold);
            if i == 1,
                H{i} = H_;
            else
                H{i} = H_ * H{i-1};
            end
        end
        
        %Stitch
        Auto_Stitched = StitchPlanarMulti(P5Obj.Images,H);

        %return
        P5Obj.Auto_Stitch = Auto_Stitched;
        
    end
    %--- End Auto Stitch ---%
    
    %----- Pose Estimate -----%
    % estiamtes pose using homography
    %
    % input:
    %     P5Obj - with Images field
    %     Fx - focal length in x
    %     Fy - focal length in y
    %     S  - scale factor
    % output:
    %
    function PoseEstimate(P5Obj,Fx,Fy,Cx,Cy,S)

        %Get Correspondences
        figure;
        Corr = zeros(4,4);
        imshow(P5Obj.Images{1}); hold on;
        [r,c,~] = size(P5Obj.Images{1});

        %top left
        title('mark top left');
        [x1, y1] = ginputc(1,'Color','g');
        plot(x1,y1,'go'); drawnow;
        Corr(1,:) = [x1 y1 0 0];

        %top right
        title('mark top right');
        [x1, y1] = ginputc(1,'Color','g');
        plot(x1,y1,'go'); drawnow;
        Corr(2,:) = [x1 y1 c 0];

        %bottom left
        title('mark bottom left');
        [x1, y1] = ginputc(1,'Color','g');
        plot(x1,y1,'go'); drawnow;
        Corr(3,:) = [x1 y1 0 r];

        %bottom right
        title('mark bottom right');
        [x1, y1] = ginputc(1,'Color','g');
        plot(x1,y1,'go'); drawnow;
        Corr(4,:) = [x1 y1 c r];

        %Get H
        H = ComputeHomography(Corr(:,3:4),Corr(:,1:2));
        
        %Pose
        P = ComputePose(H',Fx,Fy,Cx,Cy,S);
        
        %visualize
        DisplayPose(P);
        
    end
    %--- End Pose Estimate ---%
    
    %---------------------------------------------------------------------%
    %---------------------------- End Stitch -----------------------------%
    %---------------------------------------------------------------------%
    
    
    
        
    %---------------------------------------------------------------------%
    %---------------------------- Stitching ------------------------------%
    %---------------------------------------------------------------------%
    
    
    
    %---------------------------------------------------------------------%
    %-------------------------- end Stitching ----------------------------%
    %---------------------------------------------------------------------%
    
    
    
    
    
    %---------------------------------------------------------------------%
    %----------------------------- Display -------------------------------%
    %---------------------------------------------------------------------%
    
    %----- Display Matches -----%
    function DisplayMatches( Images, Locations, Matches)
       
        %show images
        [ rows1, cols1, d1 ] = size(Images{1});
        [ rows2, cols2, d2 ] = size(Images{2});
        Image = zeros( max(rows1,rows2), cols1+cols2+20, max(d1,d2), 'uint8');
        Image(1:rows1,1:cols1,:) = Images{1};
        Image(1:rows2,cols1+20:cols1+19+cols2,:) = Images{2};
        figure;
        imshow(Image);
        hold on;
        
        %show matches
        for i = 1: size(Locations{1},1)
            if (Matches(i) > 0)
                line([Locations{1}(i,2) Locations{2}(Matches(i),2)+cols1],[Locations{1}(i,1) Locations{2}(Matches(i),1)], 'Color', 'g');
            end
        end
        
    end
    %--- end Display Matches ---%
    
    %----- Display Inliers -----%
    function DisplayInliers( Images, Correspondences, Inliers_Indices)
        
        %show images
        [ rows1, cols1, d1 ] = size(Images{1});
        [ rows2, cols2, d2 ] = size(Images{2});
        Image = zeros( max(rows1,rows2), cols1+cols2+20, max(d1,d2), 'uint8');
        Image(1:rows1,1:cols1,:) = Images{1};
        Image(1:rows2,cols1+20:cols1+19+cols2,:) = Images{2};
        figure;
        imshow(Image);
        hold on;
        
        %show matches
        for i = 1: size(Correspondences{1},1)
            if( sum(i == Inliers_Indices) > 0)
                line([Correspondences{1}(i,1) Correspondences{2}(i,1)+cols1+20],[Correspondences{1}(i,2) Correspondences{2}(i,2)], 'Color', 'g');
            end
        end
    end
    %--- End Display Inliers ---%
    
    %----- Display Pose -----%
    function DisplayPose(P)
        
        %draw tag in scene
        figure;
        line([-.15 .15],[0 0],'LineWidth',4,'Color','b');
        hold on;
        
        %draw camera
        Rol = atan2(P(2,1),P(1,1));
        Pit = acos(P(3,3));
        Yaw = -atan2(P(1,3),P(2,3));
        T = -P(1:3,1:3)' * P(1:3,4);
        x = T(1);
        y = T(2);
        z = T(3);
        plot(x/1000,z/2000,'r.');
        draw_camera(x/1000,z/2000,Yaw,'r');
        axis([-2.5 2.5 -1 4]);
        
        function draw_camera(x,y,yaw,color)
            cam1 = [  0 -.2;  0 .2];
            cam2 = [  0  .2;  0 .2];
            cam3 = [-.2  .2; .2 .2];
            
            [Th,R] = cart2pol(cam1(1,:),cam1(2,:)); Th = Th + yaw +pi; [cam1(1,:),cam1(2,:)] = pol2cart(Th,R);
            [Th,R] = cart2pol(cam2(1,:),cam2(2,:)); Th = Th + yaw +pi; [cam2(1,:),cam2(2,:)] = pol2cart(Th,R);
            [Th,R] = cart2pol(cam3(1,:),cam3(2,:)); Th = Th + yaw +pi; [cam3(1,:),cam3(2,:)] = pol2cart(Th,R);
            
            cam1(1,:) = cam1(1,:) + x; cam1(2,:) = cam1(2,:) + y;
            cam2(1,:) = cam2(1,:) + x; cam2(2,:) = cam2(2,:) + y;
            cam3(1,:) = cam3(1,:) + x; cam3(2,:) = cam3(2,:) + y;
            
            line(cam1(1,:),cam1(2,:),'Color',color);
            line(cam2(1,:),cam2(2,:),'Color',color);
            line(cam3(1,:),cam3(2,:),'Color',color);
        end
    end
    %--- End Display Pose ---%
    
    %---------------------------------------------------------------------%
    %---------------------------------------------------------------------%
    %---------------------------------------------------------------------%
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    
    %% Helpers
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Helpers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %----- Compute Homography RANSAC -----%
    % uses ransac to find best homography for the correspondences
    %
    % input:
    %    Matches - correspondences in 2x1 cell
    %    Max_Iterations - "
    %    Inlier_Threshold - "
    % output:
    %    H - Homography Matrix
    %    Inlier_Indices - Indices corresponding to Matches that are inliers
    function [H, Inliers_Indices] = ComputeHomographyRANSAC( Matches, Max_Iterations, Inlier_Threshold)

        %book keeping
        Inliers = -99;
        Inliers_Indices = [];
        
        %loop for max iterations
        for i = 1:Max_Iterations

            %Grab random points
            Random_Samples = randperm(size(Matches{1},1));

            % Calculate the Transformation
            Random_Samples = Random_Samples(1:4);
            Trans_Test = ComputeHomography(Matches{2}(Random_Samples',:), Matches{1}(Random_Samples',:)); 
            if max(max(isnan(Trans_Test)))>=1, continue; end

            %Transform
            T = [Matches{2} ones(length(Matches{2}),1)] * Trans_Test;
            T_X = T(:,1) ./ T(:,end);
            T_Y = T(:,2) ./ T(:,end);
            
            %count inliers
            Update_Inliers=0; Inlier_I = [];
            X_Dist = (T_X - Matches{1}(:,1)).^2;
            Y_Dist = (T_Y - Matches{1}(:,2)).^2;
            for j=1:length(X_Dist)
                Residual=sqrt(X_Dist(j)+Y_Dist(j));
                if Residual <= Inlier_Threshold
                    Update_Inliers=Update_Inliers+1;
                    Inlier_I = [Inlier_I j];
                end
            end

            %Check if new T is better than old by inliers
            if Update_Inliers > Inliers
                Inliers   = Update_Inliers;
                Inliers_Indices = Inlier_I;
                H = Trans_Test;
            end
        end
        
        
    end
    %--- end Compute Homography RANSAC ---%
    
    %----- Compute Homography -----%
    % DLT
    %
    % input:
    %    Image1_Points - Nx2 set of points for image 1
    %    Image2_Points - Nx2 set of points for image 2
    %    Normalize - 1 = yes
    % output:
    %    H - Homography matrix
    function H = ComputeHomography( Image1_Points, Image2_Points, Normalize)

        if nargin < 3, Normalize = 0; end
        
        %nromalize
        if Normalize == 1,
            
            %0 shift
            F1 = eye(3); F2 = F1; 
            F1(1:2,end) = -mean(Image1_Points);
            F2(1:2,end) = -mean(Image2_Points);
            
            %scale
            S1 = eye(3); S2 = S1;
            S1(1,1) = 1 / std(Image1_Points(:,1));
            S1(2,2) = 1 / std(Image1_Points(:,2));
            S2(1,1) = 1 / std(Image2_Points(:,1));
            S2(2,2) = 1 / std(Image2_Points(:,2));
            
            %T
            T1 = S1 * F1;
            T2 = S2 * F2;
            Image1_Points = T1 * [Image1_Points'; ones(1,length(Image1_Points))];
            Image2_Points = T2 * [Image2_Points'; ones(1,length(Image2_Points))];
            Image1_Points = Image1_Points(1:2,:)';
            Image2_Points = Image2_Points(1:2,:)';
            
        end
        
        %A
        A = zeros(size(Image1_Points, 1)*2, 9);
        A(1:2:end,3) = 1;
        A(2:2:end,6) = 1;
        A(1:2:end,1:2) = Image1_Points;
        A(2:2:end,4:5) = Image1_Points;
        A(1:2:end,7) = -(Image2_Points(:,1) .* Image1_Points(:,1));
        A(1:2:end,8) = -(Image2_Points(:,1) .* Image1_Points(:,2));
        A(2:2:end,7) = -(Image2_Points(:,2) .* Image1_Points(:,1));
        A(2:2:end,8) = -(Image2_Points(:,2) .* Image1_Points(:,2));
        A(1:2:end,9) = -Image2_Points(:,1);
        A(2:2:end,9) = -Image2_Points(:,2);

        %SVD, Smallest singular value
        [~,~,V] = svd(A);
        
        %make H
        h = V(:,9);
        H = [ h(1) h(4) h(7); h(2) h(5) h(8); h(3) h(6) h(9) ];

        %unnormalize
        if Normalize == 1,
            H = H';
            H = T2\H*T1;
            H = H';
        end
        
        %make H(3,3) = 1
        H = H ./ H(3,3);
    end
    %--- end Compute Homography ---%
    
    %----- Compute Pose -----%
    % computes the pose from homography
    %
    % input:
    %    H - homography
    %    Fx - focal length in x
    %    Fy - focal length in y
    %    Cx - center x
    %    Cy - center y
    %    S - scale factor
    % output:
    %    P - Extrinsic Matrix [R|T]
    function P = ComputePose(H,Fx,Fy,Cx,Cy,S)
        
        %M = 3x4
        M = zeros(4,4);
        M(1,1) = (H(1,1)-Cx*H(3,1)) / Fx;
        M(1,2) = (H(1,2)-Cx*H(3,2)) / Fx;
        M(1,4) = (H(1,3)-Cx*H(3,3)) / Fx;
        M(2,1) = (H(2,1)-Cy*H(3,1)) / Fy;
        M(2,2) = (H(2,2)-Cy*H(3,2)) / Fy;
        M(2,4) = (H(2,3)-Cy*H(3,3)) / Fy;
        M(3,1) = S*H(3,1);
        M(3,2) = S*H(3,2);
        M(3,4) = S*H(3,3);
        
        %make columns on M unit vectors
        s1 = sqrt(M(1,1)^2 + M(2,1)^2 + M(3,1)^2);
        s2 = sqrt(M(1,2)^2 + M(2,2)^2 + M(3,2)^2);
        s  = sqrt(s1*s2);
        M  = M * 1/s;
        
        %force tag to be in front of camera
        if M(3,4) > 0, M = M * -1; end
        
        %third col of M by cross product
        M(1:3,3) = cross(M(1:3,1),M(1:3,2));
        
        %polar decomposition on rotation part
        R = M(1:3,1:3);
        [U,S,V] = svd(R);
        R = U*V';
        
        %T
        T = M(1:3,4);
        
        P = [R T];
    end
    %--- End Compute Pose ---%
    
    %----- Interest Points -----%
    % uses sift to find best matches
    %
    % input:
    %    Images - two images we want to match
    %    Dist   - min Dist between 1 and 2 closest distances for good match
    % output:
    %    Corr - correspondences
    function Corr = InterestPoints(Images,Dist)
        
        %variables
        Descriptors = cell(size(Images,1),1);
        Locations = cell(size(Images,1),1);
        Corr = cell(2,1);
        
        %sift
        for i=1:length(Images)
            [~,Descriptors{i},Locations{i}] = sift(Images{i});
        end  
        
        %best descriptor matches
        Matches=zeros(size(Descriptors{1},1),1);
        for i = 1 : size(Descriptors{1},1)
           [vals,inds] = sort(acos(Descriptors{1}(i,:) * Descriptors{2}'));
           if (vals(1) < Dist * vals(2)), Matches(i) = inds(1); end
        end
        
        %locations
        tmp1 = Locations{1}(Matches>0,1:2); tmp1 = tmp1(:,[2 1]);
        tmp2 = Locations{2}(nonzeros(Matches),1:2); tmp2 = tmp2(:,[2 1]);
        tmp=unique([tmp1 tmp2], 'rows');
        tmp1 = tmp(:,1:2);
        tmp2 = tmp(:,3:4);
        
        %return correspondences
        Corr{1} = tmp1;
        Corr{2} = tmp2;
        
        %display matches
        DisplayMatches(Images,Locations,Matches);
        
    end
    %--- End Interest Points ---%
    
    %----- Stitch Planar -----%
    % stitches two images together using a homography H
    %
    % input:
    %    Images - cell with 2 images
    %    H - Homography
    % output:
    %    Stitched_Image - "
    function Stitched_Image = StitchPlanar( Images, H )
            
        %image data
        Image1  = im2double(Images{1});
        Image2  = im2double(Images{2});
        [r,c,~] = size(Image2);
        R1 = Image1(:,:,1); R2 = Image2(:,:,1); 
        G1 = Image1(:,:,2); G2 = Image2(:,:,2);
        B1 = Image1(:,:,3); B2 = Image2(:,:,3);

        %mesh and transformed mesh
        H = inv(H);
        [X, Y] = meshgrid(1:c,1:r);
        [xq, yq] = meshgrid(-2000:2000,-2000:2000);
        XYq = H' * [xq(:)';yq(:)';ones(1,length(yq(:)))];
        Xq = reshape( XYq(1,:)./XYq(3,:), size(xq) );
        Yq = reshape( XYq(2,:)./XYq(3,:), size(yq) );

        %interp
        R1 = interp2(X,Y,R1,xq,yq, 'linear',0);
        G1 = interp2(X,Y,G1,xq,yq, 'linear',0);
        B1 = interp2(X,Y,B1,xq,yq, 'linear',0);
        R2 = interp2(X, Y, R2, Xq, Yq, 'linear',0);
        G2 = interp2(X, Y, G2, Xq, Yq, 'linear',0);
        B2 = interp2(X, Y, B2, Xq, Yq, 'linear',0);
        Mask1 = interp2(X,Y,ones(r,c),xq,yq,'linear',0);
        Mask2 = interp2(X,Y,ones(r,c),Xq,Yq,'linear',0);
        Vq1 = cat(3,R1,G1,B1); figure; imshow(Vq1);
        Vq2 = cat(3,R2,G2,B2); figure; imshow(Vq2);
        
        %Stitch
        Mask = Mask1+Mask2;
        Stitch = (Vq1+Vq2) ./ repmat(Mask,[1 1 3]);
        Stitch(isnan(Stitch)) = 0;
        
        %bound
        [I,J] = ind2sub(size(Mask),find(Mask==1));
        Stitched_Image = Stitch(min(I):max(I),min(J):max(J),:);
        
    end
    %--- end Stitch Planar ---%
    
    %----- Stitch Planar Multi -----%
    % stitches images together using a homographies
    %
    % input:
    %    Images - cell with multi images
    %    H - Homographies
    % output:
    %    Stitched_Image - "
    function Stitched_Image = StitchPlanarMulti( Images, HAll )
            
        %image data
        ImageD = cell(length(Images),1);
        RGB    = cell(3,length(Images));
        Mask   = cell(length(Images),1);
        Vq     = cell(length(Images),1);
        for i=1:length(Images),
            ImageD{i} = im2double(Images{i});
            RGB{1,i} = ImageD{i}(:,:,1);
            RGB{2,i} = ImageD{i}(:,:,2);
            RGB{3,i} = ImageD{i}(:,:,3);
        end
        [r,c,~] = size(ImageD{1});

        %mesh and transformed mesh
        [X, Y] = meshgrid(1:c,1:r);
        [xq, yq] = meshgrid(-1000:5000,-3000:3000);
        RGB{1,1}   = interp2(X,Y,RGB{1,1}, xq,yq,'linear',0);
        RGB{2,1}   = interp2(X,Y,RGB{2,1}, xq,yq,'linear',0);
        RGB{3,1}   = interp2(X,Y,RGB{3,1}, xq,yq,'linear',0);
        Mask{1}    = interp2(X,Y,ones(r,c),xq,yq,'linear',0);
        Vq{1}      = cat(3,RGB{1,1},RGB{2,1},RGB{3,1});
        for i=1:length(HAll),
            
            H = inv(HAll{i});
            XYq = H' * [xq(:)';yq(:)';ones(1,length(yq(:)))];
            Xq = reshape( XYq(1,:)./XYq(3,:), size(xq) );
            Yq = reshape( XYq(2,:)./XYq(3,:), size(yq) );
            
            RGB{1,i+1} = interp2(X,Y,RGB{1,i+1}, Xq,Yq, 'linear',0);
            RGB{2,i+1} = interp2(X,Y,RGB{2,i+1}, Xq,Yq, 'linear',0);
            RGB{3,i+1} = interp2(X,Y,RGB{3,i+1}, Xq,Yq, 'linear',0);
            Mask{i+1}  = interp2(X,Y,ones(r,c),  Xq,Yq,'linear',0);
            Vq{i+1}    = cat(3,RGB{1,i+1},RGB{2,i+1},RGB{3,i+1});
        end
            
        %Stitch
        Mask_F = zeros(size(Mask{1}));
        Stitch = zeros(size(Vq{1}));
        for i=1:length(Mask),
            Mask_F = Mask_F+Mask{i};
            Stitch = (Stitch+Vq{i});
        end
        Stitch = Stitch ./ repmat(Mask_F,[1 1 3]);
        Stitch(isnan(Stitch)) = 0;
        
        %bound
        [I,J] = ind2sub(size(Mask_F),find(Mask_F==1));
        Stitched_Image = Stitch(min(I):max(I),min(J):max(J),:);
        
    end
    %--- end Stitch Planar Multi ---%
    
    %----- Billboard Paste -----%
    % stitches two images together using a homography H
    %
    % input:
    %    Images - cell with 2 images
    %    H - Homography
    % output:
    %    Pasted_Image - "
    function Stitched_Image = BillboardPaste( Images, H )
            
        %image data
        Image1  = im2double(Images{1});
        Image2  = im2double(Images{2});
        [r1,c1,~] = size(Image1);
        [r2,c2,~] = size(Image2);
        R1 = Image1(:,:,1); R2 = Image2(:,:,1); 
        G1 = Image1(:,:,2); G2 = Image2(:,:,2);
        B1 = Image1(:,:,3); B2 = Image2(:,:,3);

        %mesh and transformed mesh
        H = inv(H);
        [X, Y] = meshgrid(1:c2,1:r2);
        [xq, yq] = meshgrid(-1000:1000,-1000:1000);
        XYq = H' * [xq(:)';yq(:)';ones(1,length(yq(:)))];
        Xq = reshape( XYq(1,:)./XYq(3,:), size(xq) );
        Yq = reshape( XYq(2,:)./XYq(3,:), size(yq) );

        %interp
        R1 = interp2(R1,xq,yq, 'linear',0);
        G1 = interp2(G1,xq,yq, 'linear',0);
        B1 = interp2(B1,xq,yq, 'linear',0);
        R2 = interp2(R2, Xq, Yq, 'linear',0);
        G2 = interp2(G2, Xq, Yq, 'linear',0);
        B2 = interp2(B2, Xq, Yq, 'linear',0);
        Mask1 = interp2(X,Y,ones(r1,c1),xq,yq,'linear',0);
        Mask2 = interp2(X,Y,ones(r2,c2),Xq,Yq,'linear',0);
        Vq1 = cat(3,R1,G1,B1); figure; imshow(Vq1);
        Vq2 = cat(3,R2,G2,B2); figure; imshow(Vq2);
        
        %Stitch
        Mask1 = Mask1 - Mask2;
        Stitch = Vq1.*repmat(Mask1,[1 1 3]) + Vq2.*repmat(Mask2,[1 1 3]);
        Stitch(isnan(Stitch)) = 0;
        
        %bound
        [I,J] = ind2sub(size(Mask1),find(Mask1==1));
        Stitched_Image = Stitch(min(I):max(I),min(J):max(J),:);
        
    end
    %--- end Stitch Planar ---%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end