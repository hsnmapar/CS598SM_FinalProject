[filename, pathname] = uigetfile({'*.txt'});
det = readDet([pathname '/' filename]);
val_original = det.readAll();
val_gt.all = val_original;
val_kalman.all = ExtendedKalmanFilterSize(val_original);
Q = val_kalman.all(1:3,:)'-val_gt.all;
cost.all = mean(Q(~any(isnan(Q),2),:));
for i = 1:10
    val_new = NaN(size(val_original));
    val_new(1:i:end,:) = val_original(1:i:end,:);
    val_gt.subsample_even{i} = val_new;
    val_kalman.subsample_even{i} = ExtendedKalmanFilterSize(val_new);
    Q = val_kalman.subsample_even{i}(1:3,:)'-val_gt.all;
    cost.subsample_even{i} = mean(Q(~any(isnan(Q),2),:));
end

for i = 1:10
    sel = randsample(size(val_original,1),floor(size(val_original,1)/8));
    val_new = val_original;
    val_new(sel,:) = NaN;
    val_gt.subsample_random{i} = val_new;
    val_kalman.subsample_random{i} = ExtendedKalmanFilterSize(val_new);
    Q = val_kalman.subsample_random{i}(1:3,:)'-val_gt.all;
    cost.subsample_random{i} = mean(Q(~any(isnan(Q),2),:));
end