#include <iostream>
#include <string>
#include <math.h>
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"
#include "tf/transform_listener.h"
#include "tf/transform_datatypes.h"
#include "simple_grasping/shape_extraction.h"
#include "shape_msgs/SolidPrimitive.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Vector3.h"
#include "pcl_ros/transforms.h"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/filters/crop_box.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/sample_consensus/method_types.h"
#include "pcl/sample_consensus/model_types.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/segmentation/extract_clusters.h"
#include "pcl/common/angles.h"
#include "pcl/common/common.h"
#include "pcl/point_types.h"

# define M_PI       3.14159265358979323846  /* pi */

typedef pcl::PointXYZRGB PointC;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudC;

////////////////////////////////////////////////////////////////////////////////

// Segments out a planer surface like the surface of a table.
void SegmentSurface(PointCloudC::Ptr cloud, pcl::PointIndices::Ptr indices) {

    pcl::PointIndices indices_internal;
    pcl::SACSegmentation<PointC> seg;
    seg.setOptimizeCoefficients(true);

    // Search for a plane perpendicular to some axis (specified below).
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);

    // Set the distance to the plane for a point to be an inlier.
    double distance_thresh;
    // Higher distance_thresh means more points will be included in the bounding 
    // box above and below the table top surface.
    ros::param::param("distance_thresh", distance_thresh, 0.01);
//     ros::param::param("distance_thresh", distance_thresh, 0.0075);
    seg.setDistanceThreshold(distance_thresh);
    seg.setInputCloud(cloud);

    // Make sure that the plane is perpendicular to Z-axis, 10 degree tolerance.
    Eigen::Vector3f axis;
    axis << 0, 0, 1;
    //axis << 1, 0, 0;  // To find a plane perpendicular to X-axis.
    seg.setAxis(axis);
    seg.setEpsAngle(pcl::deg2rad(10.0));

    // coeff contains the coefficients of the plane:
    // ax + by + cz + d = 0
    pcl::ModelCoefficients coeff;
    seg.segment(indices_internal, coeff);

    *indices = indices_internal;

    if (indices->indices.size() == 0) {
        ROS_ERROR("Unable to find surface.");
        return;
    }
}
    
////////////////////////////////////////////////////////////////////////////////

// Segments out a line from the point cloud.
void SegmentLine(PointCloudC::Ptr cloud, pcl::PointIndices::Ptr indices) {

    pcl::PointIndices indices_internal;
    pcl::SACSegmentation<PointC> seg;
    seg.setOptimizeCoefficients(true);

    // Search for a plane perpendicular to some axis (specified below).
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);

    // Set the distance to the plane for a point to be an inlier.
    double distance_thresh;
    // Higher distance_thresh means more points will be included in the bounding 
    // box above and below the table top surface.
    ros::param::param("distance_thresh", distance_thresh, 0.005);
//     ros::param::param("distance_thresh", distance_thresh, 0.0075);
    seg.setDistanceThreshold(distance_thresh);
    seg.setInputCloud(cloud);

    // Make sure that the plane is perpendicular to Z-axis, 10 degree tolerance.
    Eigen::Vector3f axis;
    axis << 0, 0, 1;
    //axis << 1, 0, 0;  // To find a plane perpendicular to X-axis.
    seg.setAxis(axis);
    seg.setEpsAngle(pcl::deg2rad(10.0));

    // coeff contains the coefficients of the plane:
    // ax + by + cz + d = 0
    pcl::ModelCoefficients coeff;
    seg.segment(indices_internal, coeff);

    *indices = indices_internal;

    if (indices->indices.size() == 0) {
        ROS_ERROR("Unable to find surface.");
        return;
    }
}
    
////////////////////////////////////////////////////////////////////////////////

// Removes the points belonging to the table from a given point cloud.
void SegmentSurfaceObjects(PointCloudC::Ptr cloud, pcl::PointIndices::Ptr surface_indices,
                           std::vector<pcl::PointIndices>* object_indices) {

    pcl::ExtractIndices<PointC> extract;
    pcl::PointIndices::Ptr above_surface_indices(new pcl::PointIndices());
    extract.setInputCloud(cloud);
    extract.setIndices(surface_indices);
    extract.setNegative(true);
    extract.filter(above_surface_indices->indices);
//     ROS_INFO("There are %ld points above the table", above_surface_indices->indices.size());
  
    // Increasing the cluster_tolerance will include more nearby points in a cluster.
    double cluster_tolerance;
    int min_cluster_size, max_cluster_size;
//     ros::param::param("ec_cluster_tolerance", cluster_tolerance, 0.05);
    ros::param::param("ec_cluster_tolerance", cluster_tolerance, 0.2);
    ros::param::param("ec_min_cluster_size", min_cluster_size, 10);
    ros::param::param("ec_max_cluster_size", max_cluster_size, 50000);

    // Extracting the points which belongs to the objects above the table surface.
    pcl::EuclideanClusterExtraction<PointC> euclid;
    euclid.setInputCloud(cloud);
    euclid.setIndices(above_surface_indices);
    euclid.setClusterTolerance(cluster_tolerance);
    euclid.setMinClusterSize(min_cluster_size);
    euclid.setMaxClusterSize(max_cluster_size);
    euclid.extract(*object_indices);

    // Find the size of the smallest and the largest object,
    // where size = number of points in the cluster
    size_t min_size = std::numeric_limits<size_t>::max();
    size_t max_size = std::numeric_limits<size_t>::min();
    for (size_t i = 0; i < object_indices->size(); ++i) {
        size_t cluster_size = (*object_indices)[i].indices.size();
        if (cluster_size < min_size)
            min_size = cluster_size;
        if (cluster_size > max_size)
            max_size = cluster_size;
    }

//     ROS_INFO("Found %ld objects, min size: %ld, max size: %ld",
//            object_indices->size(), min_size, max_size);
}

////////////////////////////////////////////////////////////////////////////////

// Computes the indices of the point cloud which are inside the box whose min and 
// max points are given by the min_pt and max_pt.
// This is a modified version of the function 'pcl::getPointsInBox' whose source 
// code in given here: http://docs.ros.org/groovy/api/pcl/html/common_8hpp_source.html
void getPointsInsideBox(PointCloudC &cloud, Eigen::Vector4f &min_pt, 
                        Eigen::Vector4f &max_pt, PointCloudC::Ptr output_cloud)  {
    
    std::vector<int> indices_list;  // Initial empty list of indices.
    indices_list.resize (cloud.points.size ());
    int l = 0;
    
    if (cloud.is_dense)     // If the data is dense, we don't need to check for NaN.
    {
        for (size_t i = 0; i < cloud.points.size (); ++i)
        {
            // Check if the point is inside bounds
            if (cloud.points[i].x < min_pt[0] || cloud.points[i].y < min_pt[1] || 
                cloud.points[i].z < min_pt[2])
                continue;
            if (cloud.points[i].x > max_pt[0] || cloud.points[i].y > max_pt[1] || 
                cloud.points[i].z > max_pt[2])
                continue;
            indices_list[l++] = i;
        }
    }
    else    // NaN or Inf values could exist => check for them.
    {
        for (size_t i = 0; i < cloud.points.size (); ++i)
        {
            // Check if the point is invalid
            if (!pcl_isfinite (cloud.points[i].x) || !pcl_isfinite (cloud.points[i].y) || 
                !pcl_isfinite (cloud.points[i].z))
                continue;
            // Check if the point is inside bounds
            if (cloud.points[i].x < min_pt[0] || cloud.points[i].y < min_pt[1] || 
                cloud.points[i].z < min_pt[2])
                continue;
            if (cloud.points[i].x > max_pt[0] || cloud.points[i].y > max_pt[1] || 
                cloud.points[i].z > max_pt[2])
                continue;
            indices_list[l++] = i;
        }
    }
    indices_list.resize (l);
    
    // Now assigning the indices to the point cloud indices.
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());  // Indices inside box.
    inliers->indices = indices_list;
    
    // Creating a PointCloudC pointer, because the 'extract' function takes only 
    // those kind of pointers as inputs.
    PointCloudC::Ptr cloudPtr(new PointCloudC);
    *cloudPtr = cloud;
    pcl::ExtractIndices<PointC> extract;
    extract.setInputCloud(cloudPtr);
    extract.setIndices(inliers);
    extract.filter(*output_cloud);
}

////////////////////////////////////////////////////////////////////////////////

// Computes the axis-aligned bounding box of a point cloud.
void GetAxisAlignedBoundingBox(PointCloudC::Ptr cloud, geometry_msgs::Pose* pose,
                               geometry_msgs::Vector3* dimensions) {
                               
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);

    pose->position.x = (max_pt.x() + min_pt.x()) / 2;
    pose->position.y = (max_pt.y() + min_pt.y()) / 2;
    pose->position.z = (max_pt.z() + min_pt.z()) / 2;
    pose->orientation.w = 1;

    dimensions->x = max_pt.x() - min_pt.x();
    dimensions->y = max_pt.y() - min_pt.y();
    dimensions->z = max_pt.z() - min_pt.z();  
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_pickup");
    ros::NodeHandle nh;
    
    ros::Rate loop_rate(10);

////////////////////////////////////////////////////////////////////////////////
  
    // Defining all the publishers.

    // The available point head_camera_cloud_msg is in the head camera frame. 
    // That needs to be converted into base_link frame with a transform and published.
    ros::Publisher base_link_referenced_cloud_pub =
        nh.advertise<sensor_msgs::PointCloud2>("base_link_referenced_cloud", 1, true);

    // The point cloud representing the table has to be extracted from the 
    // base link referenced point cloud and published. 
    ros::Publisher table_cloud_pub =
        nh.advertise<sensor_msgs::PointCloud2>("table_cloud", 1, true);

    // This is the publisher for the marker around the table cloud.
    ros::Publisher table_marker_pub =
        nh.advertise<visualization_msgs::Marker>("visualization_marker", 1, true);

    // This is the publisher after removing the points of the table surface from
    // the point cloud. So the remaining points belongs to the surface objects.
    ros::Publisher surface_objects_cloud_pub =
        nh.advertise<sensor_msgs::PointCloud2>("surface_objects_cloud", 1, true);

    // This is the publisher after cropping the base_link_referenced_cloud at 
    // the region of the object.
    ros::Publisher crop_pub =
        nh.advertise<sensor_msgs::PointCloud2>("cropped_cloud", 1, true);

    // This is the publisher after taking out the important section of the 
    // croppped out point cloud of the object.
    ros::Publisher output_pub =
        nh.advertise<sensor_msgs::PointCloud2>("output_cloud", 1, true);

    // This is the publisher for the marker around the cropped cloud points that 
    // gives the pose of the object.
    ros::Publisher crop_marker_pub =
        nh.advertise<visualization_msgs::Marker>("visualization_marker", 1, true);

    // This is the publisher for the marker around the cropped cloud points that 
    // gives the pose of the object.
    ros::Publisher output_marker_pub =
        nh.advertise<visualization_msgs::Marker>("visualization_marker", 1, true);

//     // This is the publisher of an example marker for marking a specified point 
//     // in the point cloud.
//     ros::Publisher example_marker_pub =
//         nh.advertise<visualization_msgs::Marker>("visualization_marker", 1, true);
        
////////////////////////////////////////////////////////////////////////////////

    // Converting the head camera referenced point cloud into base link referenced point cloud.
    sensor_msgs::PointCloud2ConstPtr head_camera_cloud_msg =
    ros::topic::waitForMessage<sensor_msgs::PointCloud2>("head_camera/depth_registered/points");

    tf::TransformListener tf_listener;
    tf_listener.waitForTransform("base_link", head_camera_cloud_msg->header.frame_id,
                                 ros::Time(0), ros::Duration(5.0));    
//     std::cout << head_camera_cloud_msg->header.frame_id << std::endl;
  
    tf::StampedTransform headCamRgbOpticalFrame_to_baseLink_transform;
    try {                                                                                 
        tf_listener.lookupTransform("base_link", head_camera_cloud_msg->header.frame_id,                    
                                    ros::Time(0), headCamRgbOpticalFrame_to_baseLink_transform);                               
    } catch (tf::LookupException& e) {                                                    
        std::cerr << e.what() << std::endl;                                                 
        return 1;                                                                           
    } catch (tf::ExtrapolationException& e) {                                             
        std::cerr << e.what() << std::endl;                                                 
        return 1;                                                                           
    }                                                                                     

    sensor_msgs::PointCloud2 base_link_referenced_cloud_msg;

    // Transform the point cloud to make it relative to base_link.
    pcl_ros::transformPointCloud("base_link", headCamRgbOpticalFrame_to_baseLink_transform, 
                                 *head_camera_cloud_msg, base_link_referenced_cloud_msg);
    
    // Since base_link_referenced_cloud_msg is a sensor_msgs/PointCloud2 message,
    // we have to convert it into PointCloudC before cropping.
    PointCloudC::Ptr base_link_referenced_cloud(new PointCloudC());
    pcl::fromROSMsg(base_link_referenced_cloud_msg, *base_link_referenced_cloud);
        
////////////////////////////////////////////////////////////////////////////////
    
    // Segmenting out the table cloud.
    
    pcl::PointIndices::Ptr table_inliers(new pcl::PointIndices());
    SegmentSurface(base_link_referenced_cloud, table_inliers);

    // Extract subset of cloud containing the table into table_cloud:
    PointCloudC::Ptr table_cloud(new PointCloudC);

    pcl::ExtractIndices<PointC> extract;
    extract.setInputCloud(base_link_referenced_cloud);
    extract.setIndices(table_inliers);
    extract.filter(*table_cloud);

    // Put data from cloud variable into ros message.
    sensor_msgs::PointCloud2 table_cloud_msg;
    pcl::toROSMsg(*table_cloud, table_cloud_msg);
    
    // Creating axis aligned bounding box for the table surface.
    visualization_msgs::Marker table_marker;
    table_marker.ns = "table";
    table_marker.header.frame_id = "base_link";
    table_marker.type = visualization_msgs::Marker::CUBE;

    // Creating an axis-aligned bounding box around the table point cloud.
    GetAxisAlignedBoundingBox(table_cloud, &table_marker.pose, &table_marker.scale);
    
    // Extending the table marker scale a bit beyond the table boundary.
    table_marker.scale.x = table_marker.scale.x + 0.1;
    table_marker.scale.y = table_marker.scale.y + 0.1;
    
    table_marker.color.r = 1;  // Color of the box is red.
    table_marker.color.a = 0.3;

////////////////////////////////////////////////////////////////////////////////
    
    // Now removing the points from the cloud that belongs to the table 
    // surface and also part of the table below the surface and the back wall.
    // Only keeping the rest of the points as the surface object points.
    // This is done by taking the point cloud above the table surface only.
    
    PointCloudC::Ptr surface_objects_cloud(new PointCloudC);

    Eigen::Vector4f surface_objects_min_pt, surface_objects_max_pt;

    // Front right top corner of the table surface.
    surface_objects_min_pt(0) = table_marker.pose.position.x - table_marker.scale.x * 0.5;
    surface_objects_min_pt(1) = table_marker.pose.position.y - table_marker.scale.y * 0.5;
    surface_objects_min_pt(2) = table_marker.pose.position.z + table_marker.scale.z * 0.5;
    
    // Back left top corner of the table surface.
    surface_objects_max_pt(0) = table_marker.pose.position.x + table_marker.scale.x * 0.5;
    surface_objects_max_pt(1) = table_marker.pose.position.y + table_marker.scale.y * 0.5;
    surface_objects_max_pt(2) = 2;    // Taking all point upto 2 meters from above the table.

    pcl::CropBox<PointC> surface_object_crop;
    surface_object_crop.setInputCloud(base_link_referenced_cloud);
    surface_object_crop.setMin(surface_objects_min_pt);
    surface_object_crop.setMax(surface_objects_max_pt);
    surface_object_crop.filter(*surface_objects_cloud);
    
    // Now we convert the cropped cloud back to sensor_msgs/PointCloud2 message
    // to publish it as a ros message.
    sensor_msgs::PointCloud2 surface_objects_cloud_msg;
    pcl::toROSMsg(*surface_objects_cloud, surface_objects_cloud_msg);

////////////////////////////////////////////////////////////////////////////////

    // Segmenting out the part of the surface point cloud within the x, y, z bounds 
    // specified by the cropping parameters set by the fetch_all_functions.py file.
    
    // Cropping parameters set by the fetch_all_functions.py file that detects objects.
    double markerX, markerY, markerZ, limitX, limitY;
    std::string objectName = "noObject";
    
    while (ros::ok())  {
        // If these parameters are not set yet, then just continue through the loop.
        // For simplicity we are only checking the existance of one parameter.
        if (nh.hasParam("markerX"))  break;
        ROS_INFO("Waiting for cropping parameters...");
        ros::spinOnce();
        loop_rate.sleep();
    }

    nh.getParam("markerX", markerX);
    nh.getParam("markerY", markerY);
    nh.getParam("markerZ", markerZ);
    nh.getParam("limitX", limitX);
    nh.getParam("limitY", limitY);
    nh.getParam("objectName", objectName);
        
////////////////////////////////////////////////////////////////////////////////

    Eigen::Vector4f min_pt(markerX-limitX, markerY-limitY, markerZ-0.2, 1);
    Eigen::Vector4f max_pt(markerX+limitX, markerY+limitY, markerZ+0.2, 1);

    // Crop the base link referenced point cloud.
    PointCloudC::Ptr cropped_cloud(new PointCloudC());
    pcl::CropBox<PointC> crop;

    crop.setInputCloud(surface_objects_cloud);
    crop.setMin(min_pt);
    crop.setMax(max_pt);
    crop.filter(*cropped_cloud);
    
    // Now we convert the cropped cloud back to sensor_msgs/PointCloud2 message
    // to publish it as a ros message.
    sensor_msgs::PointCloud2 cropped_cloud_msg;
    pcl::toROSMsg(*cropped_cloud, cropped_cloud_msg);
    
    // Now take the cropped cloud and enclose it in a marker.
    visualization_msgs::Marker crop_marker;
    crop_marker.ns = "cropped_object";
    crop_marker.id = 0;
    crop_marker.header.frame_id = "base_link";
    crop_marker.type = visualization_msgs::Marker::CUBE;
    
    // We want the box to be not axis-aligned but to bound the object tightly.
    PointCloudC::Ptr tight_cropped_cloud(new PointCloudC());
    shape_msgs::SolidPrimitive tight_cropped_shape;
    geometry_msgs::Pose tight_cropped_pose;
    simple_grasping::extractShape(*cropped_cloud, *tight_cropped_cloud, tight_cropped_shape,
                                  tight_cropped_pose);
    
    crop_marker.scale.x = tight_cropped_shape.dimensions[tight_cropped_shape.BOX_X];
    crop_marker.scale.y = tight_cropped_shape.dimensions[tight_cropped_shape.BOX_Y];
    crop_marker.scale.z = tight_cropped_shape.dimensions[tight_cropped_shape.BOX_Z];
    crop_marker.pose = tight_cropped_pose;
    crop_marker.color.g = 1;
    crop_marker.color.a = 0.3;

// //     std::cout << tight_cropped_shape.dimensions[0];   // 0 is same as tight_cropped_shape.BOX_X.
    
// ////////////////////////////////////////////////////////////////////////////////
// 
//     // Finding out the orientation in terms of roll, pitch and yaw as well, as 
//     // it will be needed later.    
//     double quatx = tight_cropped_pose.orientation.x;
//     double quaty = tight_cropped_pose.orientation.y;
//     double quatz = tight_cropped_pose.orientation.z;
//     double quatw = tight_cropped_pose.orientation.w;
// 
//     tf::Quaternion q(quatx, quaty, quatz, quatw);
//     tf::Matrix3x3 m(q);
//     double roll, pitch, yaw, rollD, pitchD, yawD;
//     m.getRPY(roll, pitch, yaw);
//     
//     rollD = roll * 180 / M_PI;      // Converting to degrees.
//     pitchD = pitch * 180 / M_PI;    // Converting to degrees.
//     yawD = yaw * 180 / M_PI;        // Converting to degrees.
//         
// ////////////////////////////////////////////////////////////////////////////////

    // Now based on what object is considered, the gripping mechanism will be 
    // different.
    PointCloudC::Ptr output_cloud(new PointCloudC);
    
    if (objectName == "nuts")  {}
    if (objectName == "coins")  {}
    if (objectName == "washers")  {}
    if (objectName == "gears")  {}
    if (objectName == "emptyBin")  {}
    if (objectName == "crankArmW" || objectName == "crankArmX" || objectName == "crankShaft")
    {
// ////////////////////////////////////////////////////////////////////////////////
// 
//         // Now to have a better grip with the crank we will be focussing on the 
//         // paddle region of the crankarm. For that we have to crop out that part of
//         // its point cloud. To do that we have to create a box marker only around 
//         // that region of the crankarm. To do that we have to find out the min and 
//         // max location of that box marker.
//         
//         Eigen::Vector4f min_pt_1, max_pt_1;    
//     
//         // Finding the minpoint of the box (which is the bottom right corner).
//         // Taking the orientation into consideration using the sin and cos. 
//         // We only have to consider the yaw as pitch and roll are 0.
//         min_pt_1(0) = crop_marker.pose.position.x - 0.5 * crop_marker.scale.x;
//         min_pt_1(1) = crop_marker.pose.position.y - 0.5 * crop_marker.scale.y;
//         min_pt_1(2) = crop_marker.pose.position.z - 0.5 * crop_marker.scale.z;
//         
//         // Finding the maxpoint of the box (which is top left corner at the center 
//         // section of the box).
//         max_pt_1(0) = min_pt_1(0) + crop_marker.scale.x;
//         max_pt_1(1) = min_pt_1(1) + crop_marker.scale.y;
//         max_pt_1(2) = min_pt_1(2) + crop_marker.scale.z;
//         
//         // Now taking out the part of the cropped point cloud which is present within 
//         // this box defined by the min and max points created above.
//         PointCloudC::Ptr output_cloud(new PointCloudC);
//         getPointsInsideBox(*cropped_cloud, min_pt_1, max_pt_1, output_cloud);
//         
// ////////////////////////////////////////////////////////////////////////////////
        
        // Fitting a line in the cropped point cloud to locate the straight region 
        // of the crankarm. And then that is produced as an output cloud.

        // Now taking out the part of the cropped point cloud which is present within 
        // this box defined by the min and max points created above.
        pcl::PointIndices::Ptr line_inliers(new pcl::PointIndices());
        SegmentLine(cropped_cloud, line_inliers);

        pcl::ExtractIndices<PointC> extractLine;
        extractLine.setInputCloud(cropped_cloud);
        extractLine.setIndices(line_inliers);
        extractLine.filter(*output_cloud);
    }
    
////////////////////////////////////////////////////////////////////////////////

    // Now we convert the output cloud back to sensor_msgs/PointCloud2 message
    // to publish it as a ros message.
    sensor_msgs::PointCloud2 output_cloud_msg;
    pcl::toROSMsg(*output_cloud, output_cloud_msg);
    
    // Now take the output cloud and enclose it in a marker.
    visualization_msgs::Marker output_marker;
    output_marker.ns = "output_object";
    output_marker.id = 1;
    output_marker.header.frame_id = "base_link";
    output_marker.type = visualization_msgs::Marker::CUBE;
    
    // We want the box to be not axis-aligned but to bound the object tightly.
    PointCloudC::Ptr tight_object_cloud(new PointCloudC());
    shape_msgs::SolidPrimitive tight_object_shape;
    geometry_msgs::Pose tight_object_pose;
    simple_grasping::extractShape(*output_cloud, *tight_object_cloud, tight_object_shape,
                                  tight_object_pose);
    
    output_marker.scale.x = tight_object_shape.dimensions[tight_object_shape.BOX_X];
    output_marker.scale.y = tight_object_shape.dimensions[tight_object_shape.BOX_Y];
    output_marker.scale.z = tight_object_shape.dimensions[tight_object_shape.BOX_Z];
    output_marker.pose = tight_object_pose;
    output_marker.color.b = 1;
    output_marker.color.a = 0.5;

// ////////////////////////////////////////////////////////////////////////////////
//     
//     // Putting an example marker before cropping.
// 
//     visualization_msgs::Marker example_marker;
//     example_marker.ns = "example";
//     example_marker.id = 111;
//     example_marker.header.frame_id = "base_link";
//     example_marker.type = visualization_msgs::Marker::CUBE;
//     
//     example_marker.pose.position.x = tight_object_pose.position.x;
//     example_marker.pose.position.y = tight_object_pose.position.y;
//     example_marker.pose.position.z = tight_object_pose.position.z;
//     example_marker.pose.orientation.x = 0.0;
//     example_marker.pose.orientation.y = 0.0;
//     example_marker.pose.orientation.z = 0.0;
//     example_marker.pose.orientation.w = 1.0;
//     example_marker.scale.x = 0.025;  // The *2 is to extend the box in both +ve and -ve x.
//     example_marker.scale.y = 0.025;  // The *2 is to extend the box in both +ve and -ve y.
//     example_marker.scale.z = 0.025;
//     example_marker.color.r = 1;
//     example_marker.color.a = 0.5;
//     
// ////////////////////////////////////////////////////////////////////////////////
    
    // Now fetch_all_functions.py file also created some parameters for this file
    // to store the values of the pose and orientation of this marker, so that 
    // these values are available to the fetch_all_functions.py file after this 
    // cpp file stops executing. Updating those parameters with the marker values.
    nh.setParam(objectName + "_posiX", crop_marker.pose.position.x);
    nh.setParam(objectName + "_posiY", crop_marker.pose.position.y);
    nh.setParam(objectName + "_posiZ", crop_marker.pose.position.z);
    nh.setParam(objectName + "_oriX", crop_marker.pose.orientation.x);
    nh.setParam(objectName + "_oriY", crop_marker.pose.orientation.y);
    nh.setParam(objectName + "_oriZ", crop_marker.pose.orientation.z);
    nh.setParam(objectName + "_oriW", crop_marker.pose.orientation.w);
    nh.setParam(objectName + "_scaleX", crop_marker.scale.x);
    nh.setParam(objectName + "_scaleY", crop_marker.scale.y);
    nh.setParam(objectName + "_scaleZ", crop_marker.scale.z);
    ROS_INFO("Set storing parameters to with marker details...");

////////////////////////////////////////////////////////////////////////////////

//    for (int i=0; i<5 and ros::ok; i++) {
    while (ros::ok)  {
        // The base_link_referenced_cloud is of type sensor_msgs/PointCloud2, so we
        // can directly publish it as a ros message using the publisher.
        base_link_referenced_cloud_pub.publish(base_link_referenced_cloud_msg);
        table_cloud_pub.publish(table_cloud_msg);
        table_marker_pub.publish(table_marker);
        surface_objects_cloud_pub.publish(surface_objects_cloud_msg);
        crop_pub.publish(cropped_cloud_msg);
        crop_marker_pub.publish(crop_marker);
        output_pub.publish(output_cloud_msg);
        output_marker_pub.publish(output_marker);
//         example_marker_pub.publish(example_marker);
        ROS_INFO("Publishing...\n");
        ROS_INFO("markerX: %f ", markerX);
        ROS_INFO("markerY: %f ", markerY);
        ROS_INFO("markerZ: %f ", markerZ);
        ROS_INFO("limitX: %f ", limitX);
        ROS_INFO("limitY: %f ", limitY);
        ROS_INFO_STREAM("objectName: " << objectName);
//         ROS_INFO("Roll: %f, Pitch: %f, Yaw: %f (all in degrees)", rollD, pitchD, yawD);
        ROS_INFO("Cropped to %ld points", cropped_cloud->size());
        ROS_INFO_STREAM("Marker details: " << std::endl << output_marker.pose << 
                        "scale: \n"<< output_marker.scale);
        
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
  
}

////////////////////////////////////////////////////////////////////////////////
