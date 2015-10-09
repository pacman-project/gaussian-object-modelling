% testing Gaussian Process to reconstruct 
% of 3D implicit funtions;
% The code use the external function plot3_to_surf
%
clear all
close all
% clc
% % 
% % a = [0 0 0; 2 2 2 ;  1 0 0; 0 1 0 ; 0 0 1; 1 1 1] 
% % A =a  ;
% % B = a ;
% % [m,p1] = size(A); [n,p2] = size(B);
% % AA = sum(A.*A,2);  % column m_by_1
% % BB = sum(B.*B,2)'; % row 1_by_n
% % DD = AA(:,ones(1,n)) + BB(ones(1,m),:) - 2*A*B';
% % EE = (sqrt(DD)) ;
% % temp = 2.*EE.^3 - 3.* EE.^2 + ones(size(EE)) 
% % 
% % return
% Building the (unknown!!) object
points_object = 50 ;
[beta_surf, lambda_surf] = ...
               meshgrid(linspace(-pi/2 ,  pi/2 ,points_object ),...
               linspace(-pi ,  pi ,points_object ) ) ;    
%meshgrid([-pi/2 : .1 : pi/2 ,-pi/2 ],  [-pi : .1 : pi, -pi] ) ;
beta = reshape(beta_surf, size(beta_surf,1) *size(beta_surf,2),1);
lambda = reshape(lambda_surf, size(lambda_surf,1) *size(lambda_surf,2),1);
%
a = 2 ;
b = .8 ;
c = .8 ;
x1_object = a .* cos(beta) .* cos(lambda) ;
x2_object = b .* cos(beta) .* sin(lambda) ;
x3_object = c .* sin(beta) ;
%
% figure
% plot3(x1_object, x2_object, x3_object,'*b','MarkerSize',1) ;
% axis equal
% grid on
%
x1_surf = reshape(x1_object,size(beta_surf)) ;
x2_surf = reshape(x2_object,size(beta_surf)) ;
x3_surf = reshape(x3_object,size(beta_surf)) ;
%
% h= surf(x1_surf, x2_surf, x3_surf ) ;
% set(h,'FaceColor', 'b','FaceAlpha',.3,'EdgeAlpha',.1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% External Sphere
a = 5 ;
b = 5 ;
c = 5 ;
x1_ext = a .* cos(beta) .* cos(lambda) ;
x2_ext = b .* cos(beta) .* sin(lambda) ;
x3_ext = c .* sin(beta) ;
%
x1_surf_ext = reshape(x1_ext,size(beta_surf)) ;
x2_surf_ext = reshape(x2_ext,size(beta_surf)) ;
x3_surf_ext = reshape(x3_ext,size(beta_surf)) ;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Internal Point
x1_int = 0 ;
x2_int = 0 ;
x3_int = 0 ;
y_int = -1*ones(size(x1_int))  ; 
%
% figure
% h= surf(x1_surf_ext, x2_surf_ext, x3_surf_ext ) ;
% set(h,'FaceColor', 'g','FaceAlpha',.2,'EdgeAlpha',.1);
% hold on
% h1= surf(x1_surf, x2_surf, x3_surf ) ;
% set(h1,'FaceColor', 'b','FaceAlpha',.3,'EdgeAlpha',.1);
% plot3(x1_int, x2_int, x3_int, '*k','MarkerSize',7) ;
% grid on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sampling the object
n_samples_obj = 20 ;
indeces_obj =  round( size(x1_object,1)*rand(1,n_samples_obj) ) ;
indeces_obj(indeces_obj<=0) = 0 ;
indeces_obj(indeces_obj>=size(x1_object,1) ) = size(x1_object,1) ;
x1_samp_obj = x1_object( indeces_obj ) ;
x2_samp_obj = x2_object( indeces_obj ) ;
x3_samp_obj = x3_object( indeces_obj ) ;
y_samp_obj = zeros(size(x1_samp_obj)) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sampling the external sphere
n_samples_ext = 20 ;
indeces_ext =  round( size(x1_ext,1)*rand(1,n_samples_ext) ) ;
indeces_ext(indeces_ext<=0) = 1 ;
indeces_ext(indeces_ext>=size(x1_ext,1) ) = size(x1_ext,1) ;
x1_samp_ext = x1_ext( indeces_ext ) ;
x2_samp_ext = x2_ext( indeces_ext ) ;
x3_samp_ext = x3_ext( indeces_ext ) ;
y_samp_ext = zeros(size(x1_samp_ext)) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
h= surf(x1_surf_ext, x2_surf_ext, x3_surf_ext ) ;
set(h,'FaceColor', 'g','FaceAlpha',.2,'EdgeAlpha',.1);
hold on
h1= surf(x1_surf, x2_surf, x3_surf ) ;
set(h1,'FaceColor', 'b','FaceAlpha',.3,'EdgeAlpha',.1);
plot3(x1_samp_obj, x2_samp_obj, x3_samp_obj, '*k','MarkerSize',7) ;
plot3(x1_samp_ext, x2_samp_ext, x3_samp_ext, '*k','MarkerSize',7) ;
plot3(x1_int, x2_int, x3_int, '*k','MarkerSize',7) ;
grid on
axis square
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1_training = [ x1_samp_obj ; x1_int  ; x1_samp_ext  ] ;
x2_training = [ x2_samp_obj ; x2_int  ; x2_samp_ext  ] ;
x3_training = [ x3_samp_obj ; x3_int  ; x3_samp_ext  ] ;
x_training = [ x1_training , x2_training , x3_training ]' ;
y_training  = [ y_samp_obj  ;  y_int  ;  y_samp_ext  ] ; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defining kernel functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
sigma_f = 1 ; % hyper-parameter of the squared exponential kernel
leng =  1 ;   % hyper-parameter of the squared exponential kernel
kernel_gauss= @(p_1,p_2) ...
              sigma_f^2*exp( ((p_1-p_2)'*(p_1-p_2))^2 /(-2*leng^2));
%
kernel_laplace = @(p_1,p_2) ...
              sigma_f^2*exp(sqrt((p_1-p_2)'*(p_1-p_2))/(-leng));
          
kernel_inv_mult = @(p_1,p_2) ... % inverse multiquadic kernel
              1/ sqrt((p_1-p_2)'*(p_1-p_2)+leng) ;  %sigma_f^2*exp(sqrt((p_1-p_2)'*(p_1-p_2))/(-leng));
%
kernel_matern_32 = @(p_1,p_2) ... %  Matèrn function with v = 3/2
             sigma_f^2 *(1+sqrt(3)* sqrt(  (p_1-p_2)'*(p_1-p_2))/leng)*...
                   exp(-sqrt(3)* sqrt(  (p_1-p_2)'*(p_1-p_2))/leng    )  ;
%
kernel_matern_52 = @(p_1,p_2) ...  %  Matèrn function with v = 5/2
             sigma_f^2 *(1+sqrt(5)* sqrt(  (p_1-p_2)'*(p_1-p_2))/leng  +    (  5*(p_1-p_2)'*(p_1-p_2))/(3*leng^2) )*...
                   exp(-sqrt(5)* sqrt(  (p_1-p_2)'*(p_1-p_2))/leng    )  ;
%
kernel_thin_plate = @(p_1,p_2) ... % thin plate function
             2*(sqrt((p_1-p_2)'*(p_1-p_2)))^3 - 3*leng*(p_1-p_2)'*(p_1-p_2) + leng^3 ;                    ;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
points = 20 ;    % sampling points (side of a cube)
[pred_x1, pred_x2, pred_x3] = ...
    meshgrid( linspace( (min(x1_samp_obj)-0.5) ,  (max(x1_samp_obj)+0.5) , points ) ,...
              linspace( (min(x2_samp_obj)-0.5) ,  (max(x2_samp_obj)+0.5), points ) ,...
              linspace( (min(x3_samp_obj)-0.5) ,  (max(x3_samp_obj)+0.5) , points ) ) ; %-1.8:.3:1.8, -1.8:.3:1.8) ;
%
pred_x1 = reshape(pred_x1, size(pred_x1,1) *size(pred_x1,2)*size(pred_x1,3),1);
pred_x2 = reshape(pred_x2, size(pred_x2,1) *size(pred_x2,2)*size(pred_x2,3),1);
pred_x3 = reshape(pred_x3, size(pred_x3,1) *size(pred_x3,2)*size(pred_x3,3),1);
%
% Adding the sampling points
pred_x1 = [pred_x1 ; x1_samp_obj ] ;
pred_x2 = [pred_x2 ; x2_samp_obj ] ;
pred_x3 = [pred_x3 ; x3_samp_obj ] ;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Regression
x_to_predict = [ pred_x1, pred_x2, pred_x3 ]' ; 
%
x_post = [ x_training , x_to_predict ] ;
Cov_tot = zeros(length(x_post),length(x_post)) ;
% tic
% for i=1:length(x_post)
%     for j=i:length(x_post)
%        Cov_tot(i,j) = kernel_gauss(x_post(:,i),x_post(:,j));
%        Cov_tot(i,j) = kernel_laplace(x_post(:,i),x_post(:,j));
%        Cov_tot(i,j) = kernel_inv_mult(x_post(:,i),x_post(:,j));
%        Cov_tot(i,j) = kernel_matern_32(x_post(:,i),x_post(:,j));
%        Cov_tot(i,j) = kernel_matern_52(x_post(:,i),x_post(:,j));
%         Cov_tot(i,j) = kernel_thin_plate(x_post(:,i),x_post(:,j));
%     end
% end
% 
% Cov_tot = Cov_tot+triu(Cov_tot,1)'; 
% toc
tic
A = x_post'  ;
B = x_post' ;
[m,p1] = size(A); [n,p2] = size(B);
AA = sum(A.*A,2);  % column m_by_1
BB = sum(B.*B,2)'; % row 1_by_n
DD = AA(:,ones(1,n)) + BB(ones(1,m),:) - 2*A*B';
EE = (sqrt(DD)) ;
temp = 2.*EE.^3 - 3.*(leng).* EE.^2 + (leng*ones(size(EE))).^3    ;
toc
Cov_tot = temp ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cut = 0.1 ;
y_post = mean_post ;
Cov_plot = Cov_post ;
variance_plot  = real(sqrt(diag(Cov_plot)));

upper_var_plot =  1.96*variance_plot ;


% y_post ;
y_post(y_post>=cut) = nan ;
y_post(y_post<=-cut) = nan ;
% y_post ;
y_post(y_post >= -cut) = 1 ;

x_post_plot = x_to_predict' ;
x_post_plot(:,1) = x_post_plot(:,1).*y_post ;
x_post_plot(:,2) = x_post_plot(:,2).*y_post ;
x_post_plot(:,3) = x_post_plot(:,3).*y_post ;

figure
h= surf(x1_surf_ext, x2_surf_ext, x3_surf_ext ) ;
set(h,'FaceColor', 'g','FaceAlpha',.2,'EdgeAlpha',.1);
hold on
h1= surf(x1_surf, x2_surf, x3_surf ) ;
set(h1,'FaceColor', 'b','FaceAlpha',.3,'EdgeAlpha',.1);
plot3(x1_samp_obj, x2_samp_obj, x3_samp_obj, 'ok','MarkerSize',7) ;
plot3(x1_samp_ext, x2_samp_ext, x3_samp_ext, '*k','MarkerSize',7) ;
plot3(x1_int, x2_int, x3_int, '*k','MarkerSize',7) ;
plot3(x_post_plot(:,1) , x_post_plot(:,2) , x_post_plot(:,3) , '*r','MarkerSize',3) ;
% s3 = scatter3(x_post_plot(:,1) , x_post_plot(:,2) , x_post_plot(:,3), 15, upper_var_plot ,'filled' )
% colorbar
grid on
axis equal
%


figure
%h= surf(x1_surf_ext, x2_surf_ext, x3_surf_ext ) ;
%set(h,'FaceColor', 'g','FaceAlpha',.2,'EdgeAlpha',.1);
hold on
h1= surf(x1_surf, x2_surf, x3_surf ) ;
set(h1,'FaceColor', 'b','FaceAlpha',.3,'EdgeAlpha',.1);
plot3(x1_samp_obj, x2_samp_obj, x3_samp_obj, 'ok','MarkerSize',7) ;
% plot3(x1_samp_ext, x2_samp_ext, x3_samp_ext, '*k','MarkerSize',7) ;
plot3(x1_int, x2_int, x3_int, '*k','MarkerSize',7) ;
% plot3(x_post_plot(:,1) , x_post_plot(:,2) , x_post_plot(:,3) , '*r','MarkerSize',3) ;
s3 = scatter3(x_post_plot(:,1) , x_post_plot(:,2) , x_post_plot(:,3), 15, upper_var_plot ,'filled' )
colorbar
grid on
axis equal


figure
plot3(x1_samp_obj, x2_samp_obj, x3_samp_obj, 'ok','MarkerSize',15) ;
hold on
plot3(x_post_plot(:,1) , x_post_plot(:,2) , x_post_plot(:,3) , '*r','MarkerSize',7) ;
s3 = scatter3(x_post_plot(:,1) , x_post_plot(:,2) , x_post_plot(:,3), 30, upper_var_plot ,'filled' )
colorbar
grid on
axis equal
%
figure 
s3 = scatter3(x_post_plot(:,1) , x_post_plot(:,2) , x_post_plot(:,3), 30, upper_var_plot ,'filled' )
colorbar% 
set(s3,'Marker','o','LineWidth',0.75); %, 'b','FaceAlpha',.3,'EdgeAlpha',.1);
axis equal
 


