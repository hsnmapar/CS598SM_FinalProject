function[] = track_tag(detection_dir) 

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

%% isolate relevant tracked features in the first frame:
[x,y, val] = klt_read_featuretable(fullfile(detection_dir, sprintf('Out_B5_N%d_R', N), 'features.txt'));

first_det = cached_dets{first}.det;
hull = round([first_det(4:5); first_det(6:7); first_det(8:9); first_det(10:11)]);

relevant_pnts = inhull([round(x(:,first)), round(y(:,first))], hull);

[rad cntr] = get_circle([x(relevant_pnts,first), y(relevant_pnts,first)]);
im = load_current_image(detection_dir, cached_dets{first}.name);
im = imshow(im);
hold on;
plot_circle(cntr(1), cntr(2), rad);
drawnow;
% now follow those points until you've lost them

for i = first_det+1:length(cached_dets)
im = load_current_image(detection_dir, cached_dets{i}.name);

im = imshow(im);
hold on;
[rad cntr] = get_circle([x(relevant_pnts,i), y(relevant_pnts,i)]);
plot_circle(cntr(1), cntr(2), rad);
drawnow;
end

keyboard;


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









