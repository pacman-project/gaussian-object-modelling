function [ h ] = plot3_to_surf( x,y,z )
%Plotting a surf strating from the x,y,z coordinates of the points
tri = delaunay(x,y);
h = trisurf(tri, x, y, z) ;
end

