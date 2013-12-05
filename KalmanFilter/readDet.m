classdef readDet < handle
    %READDET Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        line_num;
        split_s;
    end
    
    methods
        function obj = readDet(filename)
            s = fileread(filename);
            % obj.split_s = strsplit(s,'\n');
           obj.split_s = regexp(s, '\n', 'split');
            obj.line_num = 1;
        end
        
        function [val] = readline(obj)
            if(regexp(obj.split_s{obj.line_num+1}, 'loading'))
                obj.line_num = obj.line_num+1;
                val = [];
                return;
            end
            s = obj.split_s{obj.line_num+1};
            obj.line_num = obj.line_num+2;
            val = str2num(s);
        end
    end
    
end

