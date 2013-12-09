function [ C_concat ] = plot_cost( cost, fig)

C_concat = [cost.all,NaN,NaN,NaN];

for i = 2:10
    C_concat = [C_concat; cost.subsample_even{i},mean(cell2mat(cost.subsample_random(:,i)))];
end
%figure, hold on;
set(gca,'yscale','log');
hold on;
bar(C_concat);
set(gca,'XTick',[1 2:10])
set(gca,'XTickLabel',{'all';'1/2';'1/3';'1/4';'1/5';'1/6';'1/7';'1/8';'1/9';'1/10'})
legend(gca,'xerror','yerror','sizeerror','xerror random','yerror random','sizeerror random');
end

