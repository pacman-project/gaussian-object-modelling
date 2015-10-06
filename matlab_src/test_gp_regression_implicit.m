% testing Gaussian Process to reconstruct 
% of 2D implicit funtion;
% The code use the external function plot3_to_surf
%
clear all
close all
% clc
%
plot_a_priori = 0 ;
% domain in interest
%
[pred_x1 pred_x2]=  meshgrid(-2:.08:2, -2:.08:2);
pred_x1 = reshape(pred_x1, size(pred_x1,1) *size(pred_x1,2),1);
pred_x2 = reshape(pred_x2, size(pred_x2,1) *size(pred_x2,2),1);
%
pred_x = [ pred_x1 , pred_x2 ]' ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% defining a kernel
sigma_f = 1 ; % hyper-parameter of the squared exponential kernel
leng =  1 ;   % hyper-parameter of the squared exponential kernel
kernel = @(p_1,p_2) ...
              sigma_f^2*exp((p_1-p_2)'*(p_1-p_2)/(-2*leng^2));
%
% Defining the "a priori" distribution
mean_prior =  zeros(size(pred_x,2),1);
for i=1:size(pred_x,2)
variance_prior(i,1) = kernel(pred_x(:,i),pred_x(:,i)); % this is a vector corresponding to 
                  % diagonal of the covariance matrix
end
lower_prior = mean_prior-1.96*variance_prior ;
upper_prior = mean_prior+1.96*variance_prior ;
% Plotting the mean
if(plot_a_priori)
%
tri = delaunay(pred_x1(1:2:end),pred_x2(1:2:end)); % lower sampling for plotting purposes
h = trisurf(tri, pred_x1(1:2:end), pred_x2(1:2:end), mean_prior(1:2:end));
set(h,'FaceColor', 'b','FaceAlpha',.3,'EdgeAlpha',.1);
grid on
% 
hold on
% Plotting the variance
h = trisurf(tri, pred_x1(1:2:end), pred_x2(1:2:end), upper_prior(1:2:end));
set(h,'FaceColor', 'k','FaceAlpha',.01,'EdgeAlpha',.1);
grid on
h = trisurf(tri, pred_x1(1:2:end), pred_x2(1:2:end), lower_prior(1:2:end));
grid on
set(h,'FaceColor', 'k','FaceAlpha',.01,'EdgeAlpha',.1);
grid on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "A priori" distributions
%
Cov_matrix =zeros(length(pred_x1),length(pred_x2));
for i=1:length(pred_x1)
    for j=i:length(pred_x1)
        Cov_matrix(i,j)=kernel(pred_x(:,i),pred_x(:,j));
    end
end
Cov_matrix=Cov_matrix+triu(Cov_matrix,1)'; 
%
[V,D]=eig(Cov_matrix) ;
A=real(V*(D.^(1/2))) ; % A : A*A' = Cov_matrix
%
% the "a priori" distribution is assumed to be with mean_prior = zero
% and covariance matrix K = Cov_matrix, computed by the kernel
% "a priori" ~ N(mean_prior,Cov_matrix). 
% A sample of the "a priori" distribution is obtained by the following
% "a priori" ~ N(mean_prior,Cov_matrix) = A*N(0,I)+mean_prior,
% where A : A*A' = Cov_matrix
%
% For each x in the interval of interest (prediction_x)
% we randomly extract a point using the normal distribution
% probability
% Computing n samples of the "a priori" distribution
clear gaussian_process_sample;
hold on
for i=1:3
    standard_random_vector = randn(length(pred_x1),1);
    gaussian_process_sample(:,i) = A * standard_random_vector + mean_prior ; %standard_random_vector;%
    h = trisurf(tri, pred_x1(1:2:end), pred_x2(1:2:end), gaussian_process_sample((1:2:end),i));
    c = rand(1,3) ;
    set(h,'FaceColor', c,'FaceAlpha',.3,'EdgeAlpha',.1); 
    grid on
   % normal_process_sample(:,i) =  standard_random_vector;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% "A Posteriori" distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% defining the (unknown!) object to be estimated
% paremetric description of an ellipse
t = [ 0 : .1 : 2*pi ]';
scale = 1.3 ;
a = 2/scale ;
b = 0.5/scale ;
x1_object = a * cos(t) ;
x2_object = b * sin(t) ;
y_object = zeros(size(x1_object)) ;
%
figure
plot3(x1_object, x2_object, y_object,'*b','MarkerSize',7) ;
grid on
%axis equal
% taking training smaples
n_samples = 10 ;
indeces = round(length(x1_object)*rand(n_samples,1)) ;
indeces(indeces<=0) = 1 ;
x1_samp = x1_object(indeces) ;
x2_samp = x2_object(indeces) ;
y_samp = zeros(size(x1_samp)) ;
% Adding points out of the object
x1_int = 0 ;
x2_int = 0 ;
y_int = -5*ones(size(x1_int)) ;
%
n_ext = 30 ;
x1_ext = (scale*a) * cos(t) ;
x2_ext = (scale*b) * sin(t) ;
ind_ext = round(length(x1_ext)*rand(n_ext,1)) ;
ind_ext(ind_ext<=0) = 1 ;
x1_ext = x1_ext(ind_ext) ;
x2_ext = x2_ext(ind_ext) ;
y_ext = 5*ones(size(x1_ext)) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1_training = [ x1_samp ; x1_int  ; x1_ext  ];
x2_training = [ x2_samp ; x2_int  ; x2_ext  ];
y_training  = [ y_samp  ; y_int   ; y_ext  ];
%
hold on
plot3( x1_samp, x2_samp, y_samp,'*r','MarkerSize',10) ;
plot3( x1_int , x2_int , y_int,'*k','MarkerSize',5) ;
plot3( x1_ext , x2_ext , y_ext,'*k','MarkerSize',5) ;
grid on
%
% Regression on the x of interest == pred_x
x_training = [x1_training, x2_training] ;
x_to_predict = pred_x' ; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Using home-made computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  [   y_data     ]  =  [ K_dd , K_pd^T ]
%  [ y_to_predict ]  =  [ K_pd , K_pp   ]
%
% The "a posteriori" distribution is described by
% "a posteriori" ~ N(mean_post, Cov_post).
% where 
% mean_post = K_pd*inv(K_dd)*y_data ; 
% Cov_post = K_pp-K_pd*inv(K_dd)*K_pd^T ;
%
x_post = [ x_training ; x_to_predict ]' ;
Cov_tot = zeros(length(x_post),length(x_post)) ;
for i=1:length(x_post)
    for j=i:length(x_post)
        Cov_tot(i,j)=kernel(x_post(:,i),x_post(:,j));
    end
end
 % return
Cov_tot=Cov_tot+triu(Cov_tot,1)'; 
%
Cov_post_data = Cov_tot(1:length(x_training),1:length(x_training));
Cov_post_pred = Cov_tot(length(x_training)+1:end,length(x_training)+1:end);
Cov_post_pred_data = Cov_tot(length(x_training)+1:end,1:length(x_training));
% 
Cov_post_data_1 = pinv(Cov_post_data) ;
%
mean_post= Cov_post_pred_data*Cov_post_data_1*y_training ;%Cov_post_pred_data/Cov_post_data*y_training ;
Cov_post = Cov_post_pred - ...
               Cov_post_pred_data*Cov_post_data_1*Cov_post_pred_data' ;
%
h= plot3_to_surf(x_to_predict(:,1),x_to_predict(:,2), mean_post) ;
grid on
c = rand(1,3) ;
set(h,'FaceColor', c,'FaceAlpha',.3,'EdgeAlpha',.1); 
% 
%% 
for i=1:3
extimation_plot = mean_post ;
cut = (i)*0.7 ;
extimation_plot(extimation_plot>=   cut ) = nan ;
extimation_plot(extimation_plot<=  -cut ) = nan ;
extimation_plot_2 = extimation_plot ;
extimation_plot(extimation_plot<=   cut ) = 0 ;
%
% Plotting the recostruction
figure
plot3(x_to_predict(:,1),x_to_predict(:,2), extimation_plot,'*g')
hold on
plot3(x1_object, x2_object, y_object,'*b'); %,'MarkerSize',10) ;
plot3(x1_samp, x2_samp, y_samp,'*r','MarkerSize',15) ;
%
legend('estimates','real curve' ,'samples on the curve','Location','northeast')
view(0, 90)
title(['Selection filter is ',num2str(cut)])
grid on
% Computing the covariance
Cov_plot = Cov_post ;
variance_plot  = real(sqrt(diag(Cov_plot)));
lower_var_plot_2 = extimation_plot_2 - 1.96*variance_plot ;
upper_var_plot_2 = extimation_plot_2 + 1.96*variance_plot ;
lower_var_plot = extimation_plot - 1.96*variance_plot ;
upper_var_plot = extimation_plot + 1.96*variance_plot ;
color =  [ 0.4732    0.2033    1] ; %rand(1,3) ;  % [    0.3685    0.6256    0.7802]
%
% Translated Mean and Variance
figure
plot3(x_to_predict(:,1),x_to_predict(:,2), extimation_plot,'*g')
hold on
plot3(x1_object, x2_object, y_object,'*b'); %,'MarkerSize',10) ;
plot3(x1_samp, x2_samp, y_samp,'*r','MarkerSize',15) ;
plot3(x_to_predict(:,1),x_to_predict(:,2), lower_var_plot, '*','MarkerSize',3,'Color',color) ;
plot3(x_to_predict(:,1),x_to_predict(:,2), upper_var_plot, '*','MarkerSize',3,'Color', color ) ;
% 
legend('estimates','real curve' ,'samples on the curve','variance','Location','northeast')
%view(0, 90)
title(['Selection filter is ',num2str(cut)])
grid on
% Real Mean and Variance
figure
plot3(x_to_predict(:,1),x_to_predict(:,2), extimation_plot_2,'*g')
hold on
plot3(x1_object, x2_object, y_object,'*b'); %,'MarkerSize',10) ;
plot3(x1_samp, x2_samp, y_samp,'*r','MarkerSize',15) ;
plot3(x_to_predict(:,1),x_to_predict(:,2), lower_var_plot_2, '*','MarkerSize',3,'Color',color) ;
plot3(x_to_predict(:,1),x_to_predict(:,2), upper_var_plot_2, '*','MarkerSize',3,'Color', color ) ;
% 
legend('estimates','real curve' ,'samples on the curve','variance','Location','northeast')
%view(0, 90)
title(['Selection filter is ',num2str(cut)])
grid on
end


return

figure
plot(x_to_predict(:,1).*extimation_plot,x_to_predict(:,2).*extimation_plot);

% Plotting the variance
variance_post =  real(sqrt(diag(Cov_post)));
lower_post = mean_post - 1.96*variance_post ;
upper_post = mean_post + 1.96*variance_post ;
plot(prediction_x, upper_post,'k')
plot(prediction_x, lower_post,'k')
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Plotting n samples of the "a posteriori" distribution
%
% A sample of the "a posteriori" distribution is obtained by the following
% "a posteriori" ~ N(mean_post,Cov_post) = A_post*N(0,I)+mean_prior,
% where A_post : A_post*A_post' = Cov_post
%
% First we plot again previous results
figure
plot(x,y)
hold on
plot(x_samp,y_samp,'*r')
plot(prediction_x, mean_post,'--r')
plot(prediction_x, upper_post,'--k')
plot(prediction_x, lower_post,'--k')
%
[V_post,D_post]=eig(Cov_post) ;
A_post=real(V_post*(D_post.^(1/2))); % A : A*A' = Cov_matrix
%
%  n samples of the "a posteriori" distribution
for i=1:5
    standard_random_vector = randn(length(prediction_x),1);
    gaussian_process_post(:,i) = A_post * standard_random_vector  + mean_post ;
end
hold on
plot(prediction_x, gaussian_process_post,'g')


