function plotError( err )

Q = err.gt';

for i = [2,5,10]
    Q(:,end+1) = err.subsample_even{i};
    R = [];
    for j = 1:10
        R(:,end+1) = err.subsample_random{j,i};
    end
    Q(:,end+1) = mean(R');
end

plot(Q);
%set(gca, 'ymax', 100);
%set(gca,'yscale','log')
legend(gca,'all','1/2 even', '1/2 random', '1/5 even' ,'1/5 random', ...
    '1/10 even' ,'1/10 random');
% legend(gca,'all','1/2 even', '1/2 random' , '1/3 even', '1/3 random', ...
%     '1/4 even', '1/4 random', '1/5 even' ,'1/5 random', '1/6 even' ,...
%     '1/6 random' ,'1/7 even', '1/7 random' ,'1/8 even', '1/8 random', ...
%     '1/9 even', '1/9 random', '1/10 even' ,'1/10 random');

