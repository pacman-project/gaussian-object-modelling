#include <gp_node.h>

using namespace gp;

/*
 *              PLEASE LOOK at TODOs by searching "TODO" to have an idea of
 *              what is still missing or is improvable!
 */

int main (int argc, char *argv[])
{
    ros::init(argc, argv, "gaussian_process");
    GaussianProcessNode node;
    ros::Rate rate(20); //try to go at 20hz
    while (node.nh.ok())
    {
        //gogogo!
        ros::spinOnce();
        node.sampleAndPublish();
        rate.sleep();
    }
    //someone killed us :(
    return 0;
}
