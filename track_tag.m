function[err] = track_tag(detection_dir) 
% load kalman data
load(fullfile(detection_dir, 'kalman_data.mat'));

%k_all = val_kalman.all;
%% load detection info
%cached_dets = load_tag_dets(detection_dir, val_tags.all);

%types = {'all', 'subsample_even', 'subsample_random'};

cached_dets_gt = internal(detection_dir, val_kalman.all, load_tag_dets(detection_dir, val_tags.all));

err.gt = get_error(cached_dets_gt, val_tags.all);

for i = 1:length(val_kalman.subsample_even)
    cached_dets = internal(detection_dir, val_kalman.subsample_even{i}, load_tag_dets(detection_dir, val_tags.subsample_even{i}));
    err.subsample_even{i} = get_error(cached_dets, val_tags.all);
end

for i = 1:size(val_kalman.subsample_random,1)
    for j = 1:size(val_kalman.subsample_random,2)
        cached_dets = internal(detection_dir, val_kalman.subsample_random{i,j}, load_tag_dets(detection_dir, val_tags.subsample_random{i,j}));
        err.subsample_random{i,j} = get_error(cached_dets, val_tags.all);
    end
end


%% isolate relevant tracked features in the first frame:
% [x,y, val] = klt_read_featuretable(fullfile(detection_dir, sprintf('Out_B5_N%d_R', N), 'features.txt'));

% first_det = cached_dets{first}.det;
% hull = round([first_det(4:5); first_det(6:7); first_det(8:9); first_det(10:11)]);

% relevant_pnts = inhull([round(x(:,first)), round(y(:,first))], hull);

% [rad cntr] = get_circle([x(relevant_pnts,first), y(relevant_pnts,first)]);
% im = load_current_image(detection_dir, cached_dets{first}.name);
% im = imshow(im);
% hold on;
% plot_circle(cntr(1), cntr(2), rad);
% drawnow;
% % now follow those points until you've lost them

% for i = first_det+1:length(cached_dets)
% im = load_current_image(detection_dir, cached_dets{i}.name);

% im = imshow(im);
% hold on;
% [rad cntr] = get_circle([x(relevant_pnts,i), y(relevant_pnts,i)]);
% plot_circle(cntr(1), cntr(2), rad);
% drawnow;
% end

%keyboard;




function [cached_dets] = internal(detection_dir, k_all, cached_dets)
N=500;

%% find first instance of detection
first =[];
myP5 = P5_JMD();       %RANSAC
Max_Iterations = 300; %RANSAC
Inlier_Threshold = 1;  %RANSAC

for i = 1:length(cached_dets) 
    if ~isempty(cached_dets{i}.det)
        first = i;        
        break;
    end
end

[x,y, val] = klt_read_featuretable(fullfile(detection_dir, sprintf('Out_B5_N%d_R', N), 'features.txt'));
% assume first image has detection
for i = first:length(cached_dets)-1
    im = load_current_image(detection_dir, cached_dets{i}.name);
    %imshow(im);
    hold on;
    [rad, relevant_pnts] = expand_circle(k_all(i,3), k_all(i,1:2), [x(:,i), y(:,i)], val(:,i+1));
    %plot_circle(round(k_all(i,1)), round(k_all(i,2)), round(k_all(i,3)));

    if i == first % first frame, just draw the 
        det = cached_dets{i}.det;
        cached_dets{i}.hull = round([det(4:5); det(6:7); det(8:9); det(10:11)]);        
    else % compute homography from last frame
        
        
        xy_old = [x(relevant_pnts, i-1), y(relevant_pnts, i-1)];
        xy_new = [x(relevant_pnts, i), y(relevant_pnts, i)];        
        Ho = compute_homography(xy_old, xy_new);
        [H, ~] = myP5.Compute_Homography_RANSAC( {xy_new xy_old}, Max_Iterations, Inlier_Threshold);
        
        prev_hull = cached_dets{i-1}.hull;
        
        cached_dets{i}.hull = apply_homography(H', prev_hull);
        
    end
    %plot( [cached_dets{i}.hull(:,1); cached_dets{i}.hull(1,1) ], [cached_dets{i}.hull(:,2); cached_dets{i}.hull(1,2)],'g','LineWidth',4)
    %drawnow;
    %keyboard;
    hold off;
end


function[rad, relevant_pnts] = expand_circle(rad, cntr, xy, vals_next, depth)
if(nargin < 5)
    depth = 1;
end
% expand radius until we have at least 4 relevant points
num_pnts = size(xy,1);
dists = repmat(cntr, [num_pnts, 1]) - xy;
dists = sqrt(sum(dists.^2, 2));
relevant_pnts = dists <= rad;
relevant_pnts = relevant_pnts & (vals_next == 0);

if depth > 300
    rad = 1000;
    relevant_pnts = ones(size(dists)) & (vals_next == 0);
end

if length(find(relevant_pnts)) >=5 || depth > 300
    return;
else
    rad = rad+2;
    [rad relevant_pnts] = expand_circle(rad, cntr, xy, vals_next, depth+1);
end




function[H] = compute_homography(xy_old, xy_new)
% [x y; x y; x y; ...]
num_pnts = size(xy_old,1);
A = zeros(num_pnts*2, 8);
for i = 1:num_pnts
    A(2*i-1, 1:3) = [xy_old(i,:), 1];
    A(2*i-1, 7:8) = -[xy_new(i,1).*xy_old(i,:)];
    A(2*i, 4:6) = [xy_old(i,:), 1];
    A(2*i, 7:8) = -[xy_new(i,2).*xy_old(i,:)];
end
B= xy_new';
p = A\B(:);
H = ones(3,3);
H(1,:) = p(1:3);
H(2,:) = p(4:6);
H(3,1:2) = p(7:8);



function[xy_new] = apply_homography(H, xy_old);
num_pnts = size(xy_old,1);
x = [xy_old'; ones(1, num_pnts)];

tmp = H*x;
tmp(1,:) = tmp(1,:)./tmp(3,:);
tmp(2,:) = tmp(2,:)./tmp(3,:);
tmp = tmp(1:2,:);
xy_new = tmp';



function[im] = load_current_image(detection_dir, fname)
im = imread(fullfile(detection_dir,[fname,'.pnm']));
 
function plot_circle(x,y,r)
%x and y are the coordinates of the center of the circle
%r is the radius of the circle
%0.01 is the angle step, bigger values will draw the circle faster but
%you might notice imperfections (not very smooth)
ang=0:0.01:2*pi; 
xp=r*cos(ang);
yp=r*sin(ang);
plot(x+xp,y+yp);


function[rad cntr] = get_circle(pnts)
% pnts is a n by 2 x y
max_xdist =  max(pnts(:,1)) - min(pnts(:,1));
max_ydist =  max(pnts(:,2)) - min(pnts(:,2));

rad = sqrt(max_xdist^2 + max_ydist^2);
cntr = mean(pnts,1);

function [err] = get_error(cached_dets, gt)
err = [];
for i = 1:length(cached_dets)
    if(any(isnan(gt(i,:))) || ~isfield(cached_dets{i},'hull') || any(isnan(cached_dets{i}.hull(:))))
        err(i) = NaN;
        continue;
    end
    err(i) = sqrt((cached_dets{i}.hull(1,1)-gt(i,4))^2 + (cached_dets{i}.hull(1,2)-gt(i,5))^2) + ...
    sqrt((cached_dets{i}.hull(2,1)-gt(i,6))^2 + (cached_dets{i}.hull(2,2)-gt(i,7))^2) + ...
    sqrt((cached_dets{i}.hull(3,1)-gt(i,8))^2 + (cached_dets{i}.hull(3,2)-gt(i,9))^2) + ...
    sqrt((cached_dets{i}.hull(4,1)-gt(i,10))^2 + (cached_dets{i}.hull(4,2)-gt(i,11))^2);
end





