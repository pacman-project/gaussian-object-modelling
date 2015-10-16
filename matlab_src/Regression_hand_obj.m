% testing Gaussian Process to reconstruct 
% of 3D implicit funtions;
% The code use the external function plot3_to_surf
%
clear all
close all
% clc
tic
n_samples_obj = 400 ;  
n_samples_hand = 100;
n_samples_ext = 200 ; % samples on the external sphere
points = 15 ;         % side of the sampling cube
%
crop = 3 ;
% Object data
filename ='containerA_30_130.txt' ; %'containerA_30_130.txt' ; % 'obj.txt' ; %'containerA_30_130.txt' ; % 'pc_object_test.txt' ;
% filename = 'containerA_50.txt' ; % 'pc_object_test.txt' ;
% filename = 'containerA.txt' ; % 'pc_object_test.txt' ;
delimiterIn = ' ';
headerlinesIn = 15 ; % 12
% 
Object_file = importdata(filename,delimiterIn,headerlinesIn);
Points_Object = Object_file.data(:,1:3) ;
%
x1_object = Points_Object(:,1) ;
x2_object = Points_Object(:,2) ;
x3_object = Points_Object(:,3) ;
%  % To eliminate nan
x1_object = x1_object( x1_object>=min(x1_object) ) ;
x2_object = x2_object( x2_object>=min(x2_object) ) ;
x3_object = x3_object( x3_object>=min(x3_object) ) ;
% To eliminate outlier
x1_object = x1_object(crop:(end-crop)) ;
x2_object = x2_object(crop:(end-crop)) ;
x3_object = x3_object(crop:(end-crop)) ;
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % filename = 'hand.txt' ; %'containerA_30_130.txt' ; % 'obj.txt' ; %'containerA_30_130.txt' ; % 'pc_object_test.txt' ;
% % % delimiterIn = ' ';
% % % headerlinesIn = 15 ; % 12
% % % % 
% % % Hand_file = importdata(filename,delimiterIn,headerlinesIn);
% % % Points_Hand = Hand_file.data(:,1:3) ;
% % % x1_hand = Points_Hand(:,1) ;
% % % x2_hand = Points_Hand(:,2) ;
% % % x3_hand = Points_Hand(:,3) ;
% % % % % To eliminate nan
% % % x1_hand = x1_hand( x1_hand>=min(x1_hand) ) ; 
% % % x2_hand = x2_hand( x2_hand>=min(x2_hand) ) ;
% % % x3_hand = x3_hand( x3_hand>=min(x3_hand) ) ;
% % % % To eliminate outlier
% % % x1_hand = x1_hand(crop:(end-crop)) ;
% % % x2_hand = x2_hand(crop:(end-crop)) ;
% % % x3_hand = x3_hand(crop:(end-crop)) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
n_samples_obj = min(n_samples_obj, max(size(x1_object)))  ; % samples on the object
%n_samples_hand = min(n_samples_hand, max(size(x1_hand)))  ; % samples on the hand
%
x1_c = ( max(x1_object) + min(x1_object))/2 ;  
x2_c = ( max(x2_object) + min(x2_object))/2 ;  
x3_c = ( max(x3_object) + min(x3_object))/2 ;  
figure
plot3( x1_object(1:5:(end)), x2_object(1:5:(end)), x3_object(1:5:(end)), '+' )  ;
hold on
%plot3( x1_hand(1:1:(end)), x2_hand(1:1:(end)), x3_hand(1:1:(end)), '+r' )  ;
grid on
axis equal
%
% stlwrite('original.stl', x1_object(1:30:end) , x2_object(1:30:end) , x3_object(1:30:end)) ;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% External Sphere
points_object = 50 ;
[beta_surf, lambda_surf] = ...
               meshgrid(linspace(-pi/2 ,  pi/2 ,points_object ),...
               linspace(-pi ,  pi ,points_object ) ) ;    
beta = reshape(beta_surf, size(beta_surf,1) *size(beta_surf,2),1);
lambda = reshape(lambda_surf, size(lambda_surf,1) *size(lambda_surf,2),1);
x1_ext = x1_c + 0.5 .* cos(beta) .* cos(lambda) ;
x2_ext = x2_c + 0.5.* cos(beta) .* sin(lambda) ;
x3_ext = x3_c + 0.5.* sin(beta) ;
%
x1_surf_ext = reshape(x1_ext,size(beta_surf)) ;
x2_surf_ext = reshape(x2_ext,size(beta_surf)) ;
x3_surf_ext = reshape(x3_ext,size(beta_surf)) ;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Internal Point
x1_int = x1_c ;
x2_int = x2_c ;
x3_int = x3_c ;
y_int = -1*ones(size(x1_int))  ; 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sampling the object
% n_samples_obj = 50 ;
indeces_obj =  round( size(x1_object,1)*rand(1,n_samples_obj) ) ;
indeces_obj(indeces_obj<=0) = 1 ;
indeces_obj(indeces_obj>=size(x1_object,1) ) = size(x1_object,1) ;
x1_samp_obj = x1_object( indeces_obj ) ;
x2_samp_obj = x2_object( indeces_obj ) ;
x3_samp_obj = x3_object( indeces_obj ) ;
y_samp_obj = zeros(size(x1_samp_obj)) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sampling the external sphere
% n_samples_ext = 300 ;
indeces_ext =  round( size(x1_ext,1)*rand(1,n_samples_ext) ) ;
indeces_ext(indeces_ext<=0) = 1 ;
indeces_ext(indeces_ext>=size(x1_ext,1) ) = size(x1_ext,1) ;
x1_samp_ext = x1_ext( indeces_ext ) ;
x2_samp_ext = x2_ext( indeces_ext ) ;
x3_samp_ext = x3_ext( indeces_ext ) ;
y_samp_ext = ones(size(x1_samp_ext)) ; %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1_training = [ x1_samp_obj ; x1_int  ; x1_samp_ext  ] ;
x2_training = [ x2_samp_obj ; x2_int  ; x2_samp_ext  ] ;
x3_training = [ x3_samp_obj ; x3_int  ; x3_samp_ext  ] ;
x_training = [ x1_training , x2_training , x3_training ]' ;
y_training  = [ y_samp_obj  ;  y_int  ;  y_samp_ext  ] ; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sampling points (side of a cube)
% points = 15 ;   
[pred_x1, pred_x2, pred_x3] = ...
    meshgrid( linspace( (min(x1_samp_obj)-0.1*min(x1_samp_obj)) ,  (max(x1_samp_obj)+0.1*min(x1_samp_obj)) , points ) ,...
              linspace( (min(x2_samp_obj)-0.1*min(x2_samp_obj)) ,  (max(x2_samp_obj)+0.1*min(x2_samp_obj)) , points ) ,...
              linspace( (min(x3_samp_obj)-0.1*min(x3_samp_obj)) ,  (max(x3_samp_obj)+0.1*min(x3_samp_obj)) , points ) ) ; %-1.8:.3:1.8, -1.8:.3:1.8) ;
%
pred_x1 = [ reshape(pred_x1, size(pred_x1,1) *size(pred_x1,2)*size(pred_x1,3),1)  ; x1_samp_obj ] ;
pred_x2 = [ reshape(pred_x2, size(pred_x2,1) *size(pred_x2,2)*size(pred_x2,3),1) ; x2_samp_obj ] ;
pred_x3 = [ reshape(pred_x3, size(pred_x3,1) *size(pred_x3,2)*size(pred_x3,3),1) ; x3_samp_obj ] ;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Regression
x_to_predict = [ pred_x1, pred_x2, pred_x3 ]' ; 
x_post =  [ x_training , x_to_predict ] ;  % x_to_predict  ; %
%
sigma_f = 1 ; % hyper-parameter of the squared exponential kernel
leng =  1 ;   % hyper-parameter of the squared exponential kernel
  %Cov_gau   = Cov_gauss(x_post', leng, sigma_f) ;
   Cov_lap   = Cov_laplace(x_post', leng, sigma_f) ;
  % Cov_thin  = Cov_thin_plate(x_post', leng) ;
  %  Cov_nuk   = Cov_nuklei( x_post', leng ) ;
% %    Cov_mult  = Cov_inv_multiquadric(x_post', leng) ;
% %    Cov_mat_32 =Cov_matern_32(x_post', leng, sigma_f) ;
% %    Cov_mat_52 =Cov_matern_52(x_post', leng, sigma_f) ;
  %
%%
Cov_tot = Cov_lap ; % Cov_lap ;
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
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cut = 0.02 ;
y_post = mean_post ;
Cov_plot = Cov_post ;
variance_plot  = real(sqrt(diag(Cov_plot)));
upper_var_plot =  1.96*variance_plot ;
%
y_post(y_post>=cut) = nan ;
y_post(y_post<=-cut) = nan ;
y_post(y_post >= -cut) = 1 ;
%
x_post_plot = x_to_predict' ;
x1_post_plot = x_post_plot(:,1) ; 
x2_post_plot = x_post_plot(:,2) ;
x3_post_plot = x_post_plot(:,3) ;
%
x1_post_plot = x1_post_plot( y_post>-1  ) ;
x2_post_plot = x2_post_plot( y_post>-1  ) ;
x3_post_plot = x3_post_plot( y_post>-1  ) ;
y_post_plot = y_post( y_post>-1  ) ;
upper_var_plot = upper_var_plot( y_post>-1  ) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
h= surf(x1_surf_ext, x2_surf_ext, x3_surf_ext ) ;
set(h,'FaceColor', 'g','FaceAlpha',.2,'EdgeAlpha',.1);
hold on
plot3(x1_samp_obj, x2_samp_obj, x3_samp_obj, 'ok','MarkerSize',7) ;
plot3(x1_samp_ext, x2_samp_ext, x3_samp_ext, '*k','MarkerSize',7) ;
plot3(x1_int, x2_int, x3_int, '*k','MarkerSize',7) ;
plot3( x1_post_plot , x2_post_plot , x3_post_plot , '*r','MarkerSize',3) ;
grid on
axis equal
%
% figure
% plot3(x1_samp_obj, x2_samp_obj, x3_samp_obj, 'ok','MarkerSize',10) ;
% hold on
% plot3(x1_int, x2_int, x3_int, '*k','MarkerSize',100) ;
% s3 = scatter3( x1_post_plot , x2_post_plot , x3_post_plot , 30, upper_var_plot ,'filled' ) ;
% colorbar
% grid on
% axis equal
%
% tri = delaunay(x_post_plot( y_post>-1 , 1 ) , x_post_plot( y_post>-1 , 2 ) );
% figure
% h = trisurf(tri, x_post_plot( y_post>-1 , 1 ) , x_post_plot( y_post>-1 , 2 ) , x_post_plot( y_post>-1 , 3 )) ;
% set(h,'FaceColor','b','FaceAlpha',0.3,'EdgeAlpha',.1); 
% axis equal
% grid on
% hold on
% %
% stlwrite('reconstructed.stl', x1_post_plot , x2_post_plot , x3_post_plot) ;
% %
% DT = delaunayTriangulation(x_post_plot( find(y_post>-1),:)) ;
% faceColor  = [0.6875 0.8750 0.8984];
% figure
% tetramesh(DT,'FaceColor',faceColor,'FaceAlpha',0.3,'EdgeAlpha',.05);
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% max_var_post = max(upper_var_plot) ;
max_var = 0.4 ;
%
minimum = min(abs(upper_var_plot-max_var)) ;
index = find((  abs(upper_var_plot-max_var)-minimum)==0) ;
max_var_post = upper_var_plot(index) 
p_max_var_post = [ x1_post_plot(index) , x2_post_plot(index) , x3_post_plot(index) ]
n_max_var_post = unit([ (x1_c- x1_post_plot(index)),...
                   (x2_c- x2_post_plot(index)),...
                   (x3_c- x3_post_plot(index)) ])
%
%
figure
plot3(x1_samp_obj, x2_samp_obj, x3_samp_obj, 'ok','MarkerSize',10) ;
hold on
plot3(x1_int, x2_int, x3_int, '*k','MarkerSize',100) ;
s3 = scatter3( x1_post_plot , x2_post_plot , x3_post_plot , 30, upper_var_plot ,'filled' ) ;
plot3( x1_post_plot(index), x2_post_plot(index),  x3_post_plot(index), '*r','MarkerSize',70) ;
quiver3( x1_post_plot(index), x2_post_plot(index),  x3_post_plot(index),...
    0.1*n_max_var_post(1), 0.1*n_max_var_post(2), 0.1*n_max_var_post(3),'LineWidth',3 )
colorbar
grid on
axis equal



toc







