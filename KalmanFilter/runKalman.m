
%Loads the tracked tags locations and kalman filters them.  Tags are then
%subsampled randomly and evenly and then kalman filter applied.  Cost is
%the error rate for the kalman filter.

[filename, pathname] = uigetfile({'*.txt'});
det = readDet([pathname '/' filename]);
val_original = det.readAll();
val_gt.all = val_original;
val_kalman.all = ExtendedKalmanFilterSize(val_original);
Q = val_kalman.all(1:3,:)'-val_gt.all;
val_kalman.all = val_kalman.all(1:3,:)';

cost.all = mean(abs(Q(~any(isnan(Q),2),:)));
for i = 1:10
    val_new = NaN(size(val_original));
    val_new(1:i:end,:) = val_original(1:i:end,:);
    val_gt.subsample_even{i} = val_new;
    val_kalman.subsample_even{i} = ExtendedKalmanFilterSize(val_new);
    Q = val_kalman.subsample_even{i}(1:3,:)'-val_gt.all;
    cost.subsample_even{i} = mean(abs(Q(~any(isnan(Q),2),:)));
    val_kalman.subsample_even{i} = val_kalman.subsample_even{i}(1:3,:)';
end

for i = 1:10
    for j = 1:10
        sel = randsample(size(val_original,1),floor(size(val_original,1)*(j-1)/j));
        val_new = val_original;
        val_new(sel,:) = NaN;
        val_gt.subsample_random{i} = val_new;
        val_kalman.subsample_random{i,j} = ExtendedKalmanFilterSize(val_new);
        Q = val_kalman.subsample_random{i,j}(1:3,:)'-val_gt.all;
        cost.subsample_random{i,j} = mean(abs(Q(~any(isnan(Q),2),:)));
        val_kalman.subsample_random{i,j} = val_kalman.subsample_random{i,j}(1:3,:)';
    end
end