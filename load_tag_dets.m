function[cached_dets] = load_tag_dets(detection_dir)

fnames = dir(fullfile(detection_dir, '*.det'));
cached_dets = cell(length(fnames), 1);
for i = 1:length(fnames)
    fid = fopen(fullfile(detection_dir, fnames(i).name));
    tline = fgetl(fid);
    count = 1;
    cached_dets{i}.det = [];
    cached_dets{i}.name = fnames(i).name(1:end-4);

    tline = fgetl(fid);
    if ischar(tline)
        cached_dets{i}.det = str2num(tline);
    end

    fclose(fid);
end

end