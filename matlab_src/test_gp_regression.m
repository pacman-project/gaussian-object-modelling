% test
clear all
close all
use_gpml = 0 ;
use_home = 1-use_gpml;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% defining a kernel
sigma_f = 1 ; % hyper-parameter of the squared exponential kernel
leng =  1 ;   % hyper-parameter of the squared exponential kernel
kernel = @(x_1,x_2) sigma_f^2.*exp((x_1-x_2).^2/(-2*leng^2));
%
% domain in interest
prediction_x = [-2:0.01:2]';
% Defining the "a priori" distribution
mean_prior =  zeros(size(prediction_x));
variance_prior = kernel(prediction_x,prediction_x); % this is a vector corresponding to 
                  % diagonal of the covariance matrix
lower_prior = mean_prior-1.96*variance_prior ;
upper_prior = mean_prior+1.96*variance_prior ;
% Plotting the mean
p_mean_prior = plot(prediction_x,mean_prior) ;
set(p_mean_prior,'Color','k','LineWidth',1 )
hold on
% Plotting the variance
plot_variance = @(x,lower,upper,color)...
    set(fill([x,x(end:-1:1)],[upper,fliplr(lower)],color),'EdgeColor',color);
plot_variance(prediction_x,lower_prior ,upper_prior ,[0.9 0.9 0.9])
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "A priori" distributions
%
Cov_matrix =zeros(length(prediction_x),length(prediction_x));
for i=1:length(prediction_x)
    for j=i:length(prediction_x)
        Cov_matrix(i,j)=kernel(prediction_x(i),prediction_x(j));
    end
end
Cov_matrix=Cov_matrix+triu(Cov_matrix,1)'; 
%
[V,D]=eig(Cov_matrix) ;
A=real(V*(D.^(1/2))); % A : A*A' = Cov_matrix
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
standard_random_vector = randn(length(prediction_x),1) ;
gaussian_process_sample = A * standard_random_vector + mean_prior ;
%
% plotting the sample
%%%    hold on
%%%%%  plot(prediction_x, gaussian_process_sample) ;
%
% Computing n samples of the "a priori" distribution
clear gaussian_process_sample;
for i=1:7
    standard_random_vector = randn(length(prediction_x),1);
    gaussian_process_sample(:,i) = A * standard_random_vector ;
   % normal_process_sample(:,i) =  standard_random_vector;
end
hold on
plot(prediction_x, gaussian_process_sample)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% "A Posteriori" distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% defining the (unknown!) function to be estimated
x= -2:0.01:2;
y = 1*x.*(sin(x)).^2 ;
figure
plot(x,y)
% taking training smaples
n_samples = 5 ;
extra = 0 ;
samp_i = round(length(x)*rand(n_samples,1)); % 1:round((length(x)/(n_samples+2*extra))):length(x) ;
% 
%samp_i = samp_i(extra+2:end-extra-1) ;
x_samp = x(samp_i)' ;
y_samp = y(samp_i)' ;
hold on
% plotting training samples
plot(x_samp,y_samp,'*r')
%
% Regression on the x of interest 
x_to_predict = prediction_x ; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Using GPML
if(use_gpml)
meanfunc = @meanZero ; %{@meanSum, {@meanLinear, @meanConst}}; 
covfunc = @covSEiso ; 
hyp.cov =  log([ 1 ; 1]);
hyp.lik =  log(0.000001);
likfunc = @likGauss; % sn = 0.1; hyp.lik = log(sn);
%hyp = minimize(hyp, @gp, -100, @infLaplace, meanfunc, covfunc, likfunc, x_samp', y_samp');
%
[ m, s2 , fmu, fs2, lp , post] =...
    gp( hyp, @infLaplace, [], covfunc, likfunc, x_samp, y_samp, x_to_predict ) ;
%hyp2 = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, x_samp, y_samp);
%plot(z_samp,y,'*r') % real points
plot(prediction_x,m,'--g')         % estimated points
%
end
if(use_home)
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
x_post = [ x_samp ; x_to_predict ] ;
Cov_tot = zeros(length(x_post),length(x_post)) ;
for i=1:length(x_post)
    for j=i:length(x_post)
        Cov_tot(i,j)=kernel(x_post(i),x_post(j));
    end
end
Cov_tot=Cov_tot+triu(Cov_tot,1)'; 
%
Cov_post_data = Cov_tot(1:length(x_samp),1:length(x_samp));
Cov_post_pred =Cov_tot(length(x_samp)+1:end,length(x_samp)+1:end);
Cov_post_pred_data = Cov_tot(length(x_samp)+1:end,1:length(x_samp));
% 
Cov_post_data_1 = inv(Cov_post_data) ;
%
mean_post= Cov_post_pred_data/Cov_post_data*y_samp ;
Cov_post = Cov_post_pred - ...
               Cov_post_pred_data/Cov_post_data*Cov_post_pred_data' ;
%
% Plotting the mean
plot(prediction_x, mean_post,'--r')
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
end

