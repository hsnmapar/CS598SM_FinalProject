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

hold on;
plot(Q(:,1),'-k');
plot(Q(:,2),'-r');
plot(Q(:,3),'--r');
plot(Q(:,4),'-g');
plot(Q(:,5),'--g');
plot(Q(:,6),'-b');
plot(Q(:,7),'--b');
axis([0,size(Q,1),0,100]);

%set(gca, 'ymax', 100);
%set(gca,'yscale','log')
legend(gca,'all','1/2 even', '1/2 random', '1/5 even' ,'1/5 random', ...
    '1/10 even' ,'1/10 random');
% legend(gca,'all','1/2 even', '1/2 random' , '1/3 even', '1/3 random', ...
%     '1/4 even', '1/4 random', '1/5 even' ,'1/5 random', '1/6 even' ,...
%     '1/6 random' ,'1/7 even', '1/7 random' ,'1/8 even', '1/8 random', ...
%     '1/9 even', '1/9 random', '1/10 even' ,'1/10 random');

