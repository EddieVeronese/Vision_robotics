#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from detection_msgs.msg import BoundingBoxes
import numpy as np
import rospy
import open3d as o3d
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import numpy as np
import copy
from final_msgs.msg import final
from final_msgs.msg import finals




#initialization subscriber and publisher
class Node:
    def __init__(self):

        """!
        @brief Costruttore della classe Node ciao
        @param self: Istanza della classe Node.
        Inizializza l'istanza della classe Node e configura i parametri iniziali.
        @return None
        
        """

        ##variable to store the pcd
        self.pcd_file=""
        ##variable to store result of point cloud comparison 
        self.result_ransac=0
        ##variable to store the calculated box coordinates
        self.boxes = []
        ##indicates whether new information about bounding boxes has been received.
        self.flag_box=True
        ##indicates whether new information about the point cloud has been received.
        self.flag_cloud=False
        rospy.init_node('point_cloud_listener', anonymous=True)
        rospy.Subscriber("/yolov5/detections", BoundingBoxes, self.bounding_boxes_callback)
        rospy.Subscriber("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2, self.cloud_callback)
        ## This publisher is used to send messages of type finals to the 'messaggi' topic.
        self.positions_publisher = rospy.Publisher('messaggi', finals, queue_size=10)
        rospy.spin()


    

    def bounding_boxes_callback(self, data):
        """!
        @brief Callback function for bounding boxes.
        This function is called when bounding box data is received. It processes the data
        and stores the calculated box coordinates and class information in the 'boxes' attribute.
        @param data The data containing bounding box information.
        @return none
        """
        if self.flag_box==True:
            self.boxes = []
            for box in data.bounding_boxes:
                self.boxes.append(((box.xmin + box.xmax) / 2, (box.ymin + box.ymax) / 2, box.Class))
            self.flag_box=False
            self.flag_cloud=True


    #transform the center (u,v) into 3D coordinates
    def cloud_callback(self, msg):
        """!
        @brief Callback function for point cloud data.
        This function is called when new point cloud data is received. It processes the data and performs
        operations such as transformation, registration, and calculation of rotation angles. It also creates
        and publishes a message containing the processed information.
        @param msg The point cloud data received.
        """

        """!
        @var variabe to update the number of block detected
        """
        i=0; 
        if self.flag_cloud==True:

            messaggi=finals()

            #calculate the camera point for every center (u,v)
            for center in self.boxes:
                points_list = []
                u, v, classe = center
                for data in point_cloud2.read_points(msg, field_names=['x','y','z'], skip_nans=False, uvs=[(int(u), int(v))]):
                    points_list.append([data[0], data[1], data[2]])

                

                for point in points_list:

                    #matrix from camera frame to world frame
                    w_R_c = np.array([[0, -0.49948, 0.86632],
                              [-1, 0, 0],
                              [-0, -0.86632, -0.49948]])
                    x_c = np.array([-0.9, 0.24, -0.35])
                    base_offset = np.array([0.5, 0.35, 1.75])

                    pointW = np.dot(w_R_c, np.array(point)) + x_c + base_offset


                    #object object to store the filtered message
                    messaggio=final()

                    #variable to store teh filtered point cloud
                    points_list2 = []
                    for data in point_cloud2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=True):
                        x, y, z = data
                        if x > (float(point[0])-0.1) and y > (float(point[1])-0.1) and z > (float(point[2])-0.1) and x < (float(point[0])+0.1) and y < (float(point[1])+0.05) and z < (float(point[2])+0.2):
                            points_list2.append([x, y, z])
                            
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_list2)

                    #translation to centroid
                    centroid = np.mean(np.asarray(pcd.points), axis=0)
                    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) - centroid)  

                    #da camera a world
                    pcd.rotate(w_R_c)


                    self.switch_case(classe)
                    pcd_stored = o3d.io.read_point_cloud(self.pcd_file)

                    #set the voxel size for downsampling or other operations related to registration
                    voxel_size = 0.0035

                    source, target, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(
                    voxel_size, pcd_stored, pcd)

                    # execute global registration using RANSAC algorithm
                    self.result_ransac = self.execute_global_registration(source_down, target_down,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size)
                    ##refine the registration using ICP algorithm
                    self.result_icp = self.refine_registration(source, target, source_fpfh, target_fpfh,
                                    voxel_size)
                    
                    #funzioni per vedere point cloud
                    #self.visualize_point_cloud_with_axes(pcd)
                    #self.visualize_point_cloud_with_axes(pcd_stored)
                    #self.visualize_point_clouds_before_registration(source, target)
                    #self.draw_registration_result(source, target, self.result_icp.transformation)
                    


                    #matrix to calculate rotation angles
                    matrix3X3=np.array(self.result_icp.transformation[:3, :3])
                    

                    roll = np.arctan2(matrix3X3[2, 1], matrix3X3[2, 2])
                    pitch = np.arctan2(-matrix3X3[2, 0], np.sqrt(matrix3X3[2, 1]**2 + matrix3X3[2, 2]**2))
                    yaw = np.arctan2(matrix3X3[1, 0], matrix3X3[0, 0])


                    #creo messaggio
                    messaggio.classe=classe
                    messaggio.x_base=float(pointW[0])
                    messaggio.y_base=float(pointW[1])
                    messaggio.z_base=float(pointW[2])
                    messaggio.roll=roll
                    messaggio.pitch=pitch
                    messaggio.yaw=yaw
                    
                    print("Classe:", messaggio.classe)
                    print("X world:", messaggio.x_base)
                    print("Y world:", messaggio.y_base)
                    print("Z world:", messaggio.z_base)
                    print("Roll:", messaggio.roll)
                    print("Pitch:", messaggio.pitch)
                    print("Yaw:", messaggio.yaw)
                    print("-" * 30)


                    messaggi.finals.append(messaggio)
                    i =i+1
    

            messaggi.length = i
            print("Numero di blocchi trovati:", messaggi.length)
            print("-" * 30)

            self.flag_cloud=False  

            self.positions_publisher.publish(messaggi)
            rospy.signal_shutdown("Messages published")





    def refine_registration(self, source, target, source_fpfh, target_fpfh, voxel_size):
        """!
        @brief Refines the registration result using Iterative Closest Point (ICP) algorithm.
        This function takes the source and target point clouds along with their FPFH features, the voxel size used for downsampling, and the transformation obtained from global registration.
        It performs refinement of the registration result using ICP with point-to-point transformation estimation.
        The refined registration result is returned.
        @param source The source point cloud.
        @param target The target point cloud.
        @param source_fpfh The FPFH features of the source point cloud.
        @param target_fpfh The FPFH features of the target point cloud.
        @param voxel_size The voxel size used for downsampling.
        @return The refined registration result.
        """
        distance_threshold = voxel_size * 0.4
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, self.result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        return result

    

    def switch_case(self, case):
        """!
        @brief Switch case function for setting the PLY file based on block recognized.
        This function takes a case as input and assigns the corresponding PLY file path to the 'pcd_file' attribute.
        If the case does not match any of the predefined cases, it prints an error message.
        @param case The case to switch upon.
        @return None
        """

        if case == "X1-Y1-Z2":
            self.pcd_file = "/home/eddie/ros_ws/src/vision/nodes/ply/x1-y1-z2.ply"
        elif case == "X1-Y2-Z1":
            self.pcd_file = "/home/eddie/ros_ws/src/vision/nodes/ply/x1-y2-z1.ply"
        elif case == "X1-Y2-Z2":
            self.pcd_file = "/home/eddie/ros_ws/src/vision/nodes/ply/x1-y2-z2.ply"
        elif case == "X1-Y2-Z2-CHAMFER":
            self.pcd_file = "/home/eddie/ros_ws/src/vision/nodes/ply/x1-y2-z2-chamfer.ply"
        elif case == "X1-Y2-Z2-TWINFILLET":
            self.pcd_file = "/home/eddie/ros_ws/src/vision/nodes/ply/x1-y2-z2-twinfillet.ply"
        elif case == "X1-Y3-Z2":
            self.pcd_file = "/home/eddie/ros_ws/src/vision/nodes/ply/x1-y3-z2.ply"
        elif case == "X1-Y3-Z2-FILLER":
            self.pcd_file = "/home/eddie/ros_ws/src/vision/nodes/ply/x1-y3-z2-fillet.ply"
        elif case == "X1-Y4-Z1":
            self.pcd_file = "/home/eddie/ros_ws/src/vision/nodes/ply/x1-y4-z1.ply"
        elif case == "X1-Y4-Z2":
            self.pcd_file = "/home/eddie/ros_ws/src/vision/nodes/ply/x1-y4-z2.ply"
        elif case == "X2-Y2-Z2":
            self.pcd_file = "/home/eddie/ros_ws/src/vision/nodes/ply/x2-y2-z2.ply"
        elif case == "X2-Y2-Z2-FILLET":
            self.pcd_file = "/home/eddie/ros_ws/src/vision/nodes/ply/x2-y2-z2-fillet.ply"
        else:
            print("PLY file error")




    

    def preprocess_point_cloud(self, pcd, voxel_size):
        """!
        @brief Preprocesses a point cloud by downsampling and computing FPFH features.
        This function takes a point cloud and performs voxel downsampling using the specified voxel size.
        It then estimates the normals and computes Fast Point Feature Histograms (FPFH) for the downsampled point cloud.
        The downsampled point cloud and its FPFH features are returned.
        @param pcd The input point cloud to preprocess.
        @param voxel_size The voxel size for downsampling.
        @return A tuple containing the downsampled point cloud and its FPFH features.
        """

        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh




    def prepare_dataset(self, voxel_size, pcd_stored, pcd):
        """!
        @brief Prepares the dataset for global registration by downsampling and computing features.
        This function takes a point cloud and performs voxel downsampling and computes Fast Point Feature Histograms (FPFH).
        It returns the original source and target point clouds, as well as the downsampled versions and their respective FPFH features.
        @param voxel_size The voxel size for downsampling.
        @param pcd_stored The stored point cloud used as the target for registration.
        @param pcd The input point cloud to register.
        @return A tuple containing the source, target, downsampled source, downsampled target, source FPFH features, and target FPFH features.
        """

        source = copy.deepcopy(pcd)
        target = copy.deepcopy(pcd_stored)
        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh



    
    def execute_global_registration(self, source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        
        """!
        @brief Executes the global registration between downsampled point clouds using feature matching.
        This function takes the downsampled source and target point clouds, along with their corresponding FPFH features,
        and performs global registration using RANSAC-based feature matching. It returns the registration result.
        @param source_down The downsampled source point cloud.
        @param target_down The downsampled target point cloud.
        @param source_fpfh The FPFH features of the source point cloud.
        @param target_fpfh The FPFH features of the target point cloud.
        @param voxel_size The voxel size used for downsampling.
        @return The registration result.
        """
        
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result


    
    def draw_registration_result(self, source, target, transformation):
        """!
        @brief Draws the registration result by visualizing the transformed source and target point clouds.
        This function takes the source and target point clouds along with the transformation matrix and visualizes
        the transformed source and target point clouds. It also displays a coordinate frame indicating the final
        transformation.
        @param source The source point cloud before transformation.
        @param target The target point cloud before transformation.
        @param transformation The transformation matrix to apply to the source point cloud.
        @return None
        """

        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        coordinate_frame.transform(transformation)  # Applica la trasformazione finale
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                        zoom=0.559,
                                        front=[0.0, 0.0, -1.0],
                                        lookat=[0, 0, 0],
                                        up=[0, 1, 0])
        

   
    def visualize_point_clouds_before_registration(self, source, target):
        """!
        @brief Visualizes the source and target point clouds before registration.
        This function takes the source and target point clouds, translates them to the origin,
        and visualizes them with different colors. It provides a view of the point clouds
        before they are aligned.
        @param source The source point cloud before registration.
        @param target The target point cloud before registration.
        @return None
        """

        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        # Translate source point cloud to the origin
        centroid_source = np.mean(np.asarray(source_temp.points), axis=0)
        source_temp.points = o3d.utility.Vector3dVector(np.asarray(source_temp.points) - centroid_source)

        # Translate target point cloud to the origin
        centroid_target = np.mean(np.asarray(target_temp.points), axis=0)
        target_temp.points = o3d.utility.Vector3dVector(np.asarray(target_temp.points) - centroid_target)

        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])

        o3d.visualization.draw_geometries([source_temp, target_temp],
                                        zoom=0.559,
                                        front=[0.0, 0.0, -1.0],
                                        lookat=[0, 0, 0],
                                        up=[0, 1, 0])



    def visualize_point_cloud_with_axes(self, pcd):
        """!
        @brief Visualizes a single point cloud along with a coordinate frame.
        This function takes a point cloud and displays it along with a coordinate frame.
        It provides a visual representation of the point cloud with reference axes.
        @param pcd The point cloud to visualize.
        @return None
        """

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coordinate_frame],
                                        zoom=0.559,
                                        front=[0.0, 0.0, -1.0],
                                        lookat=[0, 0, 0],
                                        up=[0, 1, 0])
         

            

if __name__ == '__main__':
    node = Node()



