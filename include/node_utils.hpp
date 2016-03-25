#ifndef _NODE_UTILS_HPP_
#define _NODE_UTILS_HPP_

//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <random_generation.hpp>

/** Some utility functions to color points in point clouds */

/** \brief Color a point with specified color */
void colorIt(uint8_t r, uint8_t g, uint8_t b, pcl::PointXYZRGB &pt)
{
    uint32_t rgb = ((uint32_t)r <<16 | (uint32_t)g <<8 | (uint32_t)b);
    pt.rgb = *reinterpret_cast<float*>( &rgb );
}

void colorThem(uint8_t r, uint8_t g, uint8_t b, pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    for (auto &pt : cloud.points)
        colorIt(r,g,b, pt);
}
/** \brief Color point cloud with specified color */
void colorThem(uint8_t r, uint8_t g, uint8_t b, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    colorThem(r,g,b,*cloud);
}

#endif
