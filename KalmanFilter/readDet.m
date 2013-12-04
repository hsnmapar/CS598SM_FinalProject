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
            obj.split_s = strsplit(s,'\n');
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
        
        function [val, eof] = readCenterSizeLine(obj)
            [val, eof] = readline(obj);
            if(isempty(val))
                val = [NaN,NaN,NaN];
                return;
            end
            v = val([2,3]);
            v(3) = sqrt(max((val(4)-val(8))^2+(val(5)-val(9))^2,(val(6)-val(10))^2+(val(7)-val(11))^2));
            val = v;
        end
        
        function [vals] = readAll(obj)
           eof = 0;
           vals = [];
           while ~eof
                [val,eof] = obj.readCenterSizeLine();
                vals(end+1,:) = val;
           end
        end
    end
    
end

