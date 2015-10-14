% testing Gaussian Process to reconstruct 
% of 3D objects from point cloud data;
%
clear all
close all
%
% Importing Object Data
filename = 'containerA.txt' ; % 'pc_object_test.txt' ;
delimiterIn = ' ';
headerlinesIn = 15 ; % 12
% 
Object_file = importdata(filename,delimiterIn,headerlinesIn);
Points_Object = Object_file.data(:,1:3) ;
%
x1_object = Points_Object(:,1) ;
x2_object = Points_Object(:,2) ;
x3_object = Points_Object(:,3) ;
%
figure
plot3( x1_object(1:20:end), x2_object(1:20:end), x3_object(1:20:end), '+' )  ;
% plot3( x1_object , x2_object , x3_object , '+' )  ;
grid on
axis equal
%
% tri = delaunayTriangulation(x1_object(1:20:end),x2_object(1:20:end),x3_object(1:20:end)) ;
% figure
% tetramesh(tri,'FaceColor','g','FaceAlpha',0.3,'EdgeColor','k' ,'EdgeAlpha',.01);
% axis equal
% grid on
% 
% Sampling the Object
n_samples_obj = 200 ;
%
indeces_obj =  round( size(x1_object,1)*rand(1,n_samples_obj) ) ;
indeces_obj(indeces_obj<=0) = 0 ;
indeces_obj(indeces_obj>=size(x1_object,1) ) = size(x1_object,1) ;
%
x1_samp_obj = x1_object( indeces_obj ) ;
x2_samp_obj = x2_object( indeces_obj ) ;
x3_samp_obj = x3_object( indeces_obj ) ;
y_samp_obj = zeros(size(x1_samp_obj)) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Building the external sphere
x1_c = ( max(x1_object) - min(x1_object))/2 ;  
x2_c = ( max(x2_object) - min(x2_object))/2 ;  
x3_c = ( max(x3_object) - min(x3_object))/2 ;  
%
points_object = 50 ;
[beta_surf, lambda_surf] = ...
               meshgrid(linspace(-pi/2 ,  pi/2 ,points_object ),...
               linspace(-pi ,  pi ,points_object ) ) ;    
beta = reshape(beta_surf, size(beta_surf,1) *size(beta_surf,2),1);
lambda = reshape(lambda_surf, size(lambda_surf,1) *size(lambda_surf,2),1);
%
x1_surf_ext = x1_c + 0.04 .* cos(beta) .* cos(lambda) ;
x2_surf_ext = x2_c + 0.04 .* cos(beta) .* sin(lambda) ;
x3_surf_ext = x3_c + 0.04 .* sin(beta) ;
%
% Sampling the External Sphere
n_samples_ext = 100 ;
%
indeces_ext =  round( size(x1_surf_ext,1)*rand(1,n_samples_ext) ) ;
indeces_ext(indeces_ext<=0) = 1 ;
indeces_ext(indeces_ext>=size(x1_surf_ext,1) ) = size(x1_surf_ext,1) ;
%
x1_samp_ext = x1_surf_ext( indeces_ext ) ;
x2_samp_ext = x2_surf_ext( indeces_ext ) ;
x3_samp_ext = x3_surf_ext( indeces_ext ) ;
y_samp_ext = ones(size(x1_samp_ext)) ;
%
% Internal Points
x1_int = x1_c;
x2_int = x2_c ;
x3_int = x3_c ;
y_int = -1 *ones(size(x1_int))  ; 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1_training = [ x1_samp_obj ; x1_samp_ext  ] ;% [ x1_samp_obj ; x1_int  ; x1_samp_ext  ] ;
x2_training = [ x2_samp_obj ; x2_samp_ext  ] ;%[ x2_samp_obj ; x2_int  ; x2_samp_ext  ] ;
x3_training = [ x3_samp_obj ; x3_samp_ext  ] ;% [ x3_samp_obj ; x3_int  ; x3_samp_ext  ] ;
y_training  = [ y_samp_obj  ; y_samp_ext  ] ;  [ y_samp_obj  ;  y_int  ;  y_samp_ext  ] ; 
%
x_training = [ x1_training , x2_training , x3_training ]' ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prediction points (side of a cube)
points = 20 ;    
[pred_x1, pred_x2, pred_x3] = ...
    meshgrid( linspace( (min(x1_samp_obj)-0.005) ,  (max(x1_samp_obj)+0.005) , points ) ,...
              linspace( (min(x2_samp_obj)-0.005) ,  (max(x2_samp_obj)+0.005), points ) ,...
              linspace( (min(x3_samp_obj)-0.005) ,  (max(x3_samp_obj)+0.005) , points ) ) ; %-1.8:.3:1.8, -1.8:.3:1.8) ;
%
pred_x1 = reshape(pred_x1, size(pred_x1,1) *size(pred_x1,2)*size(pred_x1,3),1);
pred_x2 = reshape(pred_x2, size(pred_x2,1) *size(pred_x2,2)*size(pred_x2,3),1);
pred_x3 = reshape(pred_x3, size(pred_x3,1) *size(pred_x3,2)*size(pred_x3,3),1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_to_predict = [ pred_x1, pred_x2, pred_x3 ]' ; 
%
x_post =  [ x_training , x_to_predict ]   ;  %   x_training ,
%
% Computation of the covarinace matrix with the thin plate kernel
sigma_f = 1 ; % hyper-parameter of the squared exponential kernel
leng =  1 ;   % hyper-parameter of the squared exponential kernel
tic
A = x_post'  ;
B = x_post' ;
[ m, p1 ] = size(A) ;
[ n, p2 ] = size(B);
AA = sum(A.*A,2);  % column m_by_1
BB = sum(B.*B,2)'; % row 1_by_n
DD = AA(:,ones(1,n)) + BB(ones(1,m),:) - 2*A*B';
EE = (sqrt(DD)) ;
Cov_tot = 2.*EE.^3 - 3.*(leng).* EE.^2 + (leng*ones(size(EE))).^3    ;
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%  [   y_data     ]  =  [ K_dd , K_pd^T ]
%  [ y_to_predict ]  =  [ K_pd , K_pp   ]
%
% The "a posteriori" distribution is described by
% "a posteriori" ~ N(mean_post, Cov_post).
% where 
% mean_post = K_pd*inv(K_dd)*y_data ; 
% Cov_post = K_pp-K_pd*inv(K_dd)*K_pd^T ;
Cov_post_data = Cov_tot(1:length(x_training),1:length(x_training));
Cov_post_pred = Cov_tot(length(x_training)+1:end,length(x_training)+1:end);
Cov_post_pred_data = Cov_tot(length(x_training)+1:end,1:length(x_training));
% 
Cov_post_data_1 = pinv(Cov_post_data) ;
%
mean_post= Cov_post_pred_data*Cov_post_data_1*y_training ; %Cov_post_pred_data/Cov_post_data*y_training ;
Cov_post = Cov_post_pred - ...
               Cov_post_pred_data*Cov_post_data_1*Cov_post_pred_data' ;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cut = 0.001 ;
y_post = mean_post ;
Cov_plot = Cov_post ;
variance_plot  = real(sqrt(diag(Cov_plot)));
%
upper_var_plot =  1.96*variance_plot ;
%
y_post(y_post>=cut) = nan ;
y_post(y_post<=-cut) = nan ;
y_post(y_post >= -cut) = 1 ;

y_post_plot = y_post(find(y_post>=-cut) ) ;
x1_post_plot = pred_x1(find(y_post>=-cut) ) ;
x2_post_plot = pred_x2(find(y_post>=-cut) ) ;
x3_post_plot = pred_x3(find(y_post>=-cut) ) ;
figure
plot3( x1_post_plot, x2_post_plot, x3_post_plot, '+b' )  ;
% plot3( x1_object , x2_object , x3_object , '+' )  ;
grid on
axis equal
hold on
plot3( x1_samp_obj, x2_samp_obj, x3_samp_obj, '*r' , 'MarkerSize',15)  ;
grid on
axis equal
hold on
figure
plot3( x1_samp_obj, x2_samp_obj, x3_samp_obj, '*r' , 'MarkerSize',15)  ;
grid on
axis equal
hold on







