[filename, pathname] = uigetfile({'*.txt'});
det = readDet([pathname '/' filename]);
val_original = det.readAll();
val_kalman = ExtendedKalmanFilterSize(val_original);