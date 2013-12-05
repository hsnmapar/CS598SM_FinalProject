function[] = track_tag(detection_dir) 
% load kalman data
load(fullfile(detection_dir, 'kalman_data.mat'));

N=300;

%% load detection info
cached_dets = load_tag_dets(detection_dir);

%% find first instance of detection
first =[];

for i = 1:length(cached_dets) 
    if ~isempty(cached_dets{i}.det)
        first = i;        
        break;
    end
end

[x,y, val] = klt_read_featuretable(fullfile(detection_dir, sprintf('Out_B5_N%d_R', N), 'features.txt'));
% assume first image has detection
for i = 1:length(cached_dets)
    im = load_current_image(detection_dir, cached_dets{i}.name);
    imshow(im);
    hold on;
    [rad, relevant_pnts] = expand_circle(val_kalman.all(i,3), val_kalman.all(i,1:2), [x(:,i), y(:,i)]);
    if i == 1 % first frame, just draw the 
        det = cached_dets{i}.det;
        hull = round([det(4:5); det(6:7); det(8:9); det(10:11)]);
        plot(hull(:,1), hull(:,2))
        keyboard;
    else
    end
    plot_circle(round(val_kalman.all(i,1)), round(val_kalman.all(i,2)), round(val_kalman.all(i,3)));
    drawnow;
    hold off;
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

keyboard;





function[rad, relevant_pnts] = expand_circle(rad, cntr, xy)
% expand radius until we have at least 4 relevant points
num_pnts = size(xy,1);
dists = repmat(cntr, [num_pnts, 1]) - xy;
dists = sqrt(sum(dists.^2, 2));
if length(find(dists <= rad)) >=4
    relevant_pnts = dists <= rad;
else
    rad = rad+2;
    [rad relevant_pnts] = expand_circle(rad, cntr, xy);
end




function[H] = compute_homography(xy_old, xy_new)
% [x y; x y; x y; ...]
num_pnts = size(xy_old,1);
A = zeros(num_pnts*2, 8);
for i = 1:num_pnts
    A(2*i-1, 1:3) = [xy_old(i,:), 1];
    A(2*i-1, 7:8) = -[xy_new(1).*xy_old];
    A(2*i, 4:6) = [xy_old(i,:), 1];
    A(2*i-1, 7:8) = -[xy_new(2).*xy_old];
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









