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
        
        function [val, eof] = readline(obj)
            eof = 0;
            if(obj.line_num >= length(obj.split_s))
                eof = 1;
                val = [];
                return;
            end
                
            if(regexp(obj.split_s{obj.line_num+1}, 'loading'))
                obj.line_num = obj.line_num+1;
                val = [];
                return;
            end
            s = obj.split_s{obj.line_num+1};
            obj.line_num = obj.line_num+2;
            val = str2num(s);
        end
        
        function [v, val, eof] = readCenterSizeLine(obj)
            [val, eof] = readline(obj);
            if(isempty(val))
                val = NaN(1,11);
                v = [NaN,NaN,NaN];
                return;
            end
            v = val([2,3]);
            v(3) = sqrt(max((val(4)-val(8))^2+(val(5)-val(9))^2,(val(6)-val(10))^2+(val(7)-val(11))^2));
        end
        
        function [center_radius, tag_vals, def] = readAll(obj)
           eof = 0;
           center_radius = [];
           tag_vals = [];
           while ~eof
                [v_cr, v_tag, eof] = obj.readCenterSizeLine();
                tag_vals(end+1,:) = v_tag;
                center_radius(end+1,:) = v_cr;
           end
        end
    end
    
end

