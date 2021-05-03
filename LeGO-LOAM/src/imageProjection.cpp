// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "utility.h"


class ImageProjection{
private:

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    
    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;

    pcl::PointCloud<PointType>::Ptr fullCloud;                          // 该次扫描的全部点
    pcl::PointCloud<PointType>::Ptr fullInfoCloud;

    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;                     // 给下一步使用的点云
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;                 // 经过提取的点，去除了地面和无效聚类的离散点
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    PointType nanPoint;

    cv::Mat rangeMat;
    cv::Mat labelMat;
    cv::Mat groundMat;
    int labelCount;

    float startOrientation;
    float endOrientation;

    cloud_msgs::cloud_info segMsg;
    std_msgs::Header cloudHeader;

    std::vector<std::pair<uint8_t, uint8_t> > neighborIterator;

    uint16_t *allPushedIndX;
    uint16_t *allPushedIndY;

    uint16_t *queueIndX;
    uint16_t *queueIndY;

public:
    ImageProjection():
        nh("~"){
        // 订阅来自velodyne雷达驱动的topic ("/velodyne_points")
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1, &ImageProjection::cloudHandler, this);
        
        // 发布
        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        // 发布地面点云
        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();
        resetParameters();
    }

	// 初始化各类参数以及分配内存
    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<PointType>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        // 分配数组
        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

		// labelComponents函数中用到了这个矩阵
		// 该矩阵用于求某个点的上下左右4个邻接点
        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }

	// 初始化/重置各类参数内容
    void resetParameters(){
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    ~ImageProjection(){}
	
    /// 将ros msg的点云拷贝到 成员变量 laserCloudIn
    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
        // 将ROS中的sensor_msgs::PointCloud2ConstPtr类型转换到pcl点云库指针
        cloudHeader = laserCloudMsg->header;
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
    }
    
    //
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        copyPointCloud(laserCloudMsg);
        findStartEndAngle();
        // 将点云转换成矩阵
        projectPointCloud();
        // 获取地面点
        groundRemoval();
        // 使用BFS扩展搜索，进行聚类，给labelMat矩阵进行赋值
        cloudSegmentation();
        publishCloud();
        resetParameters();
    }

    // 获取一帧扫描的起始角度、结束角度
    void findStartEndAngle(){
        // 雷达坐标系：右->X,前->Y,上->Z [这是velodyne官方给的坐标系，但是ros驱动进行了改变，x轴向前]
        // 雷达内部旋转扫描方向：Z轴俯视下来，顺时针方向（Z轴右手定则反向）

        // atan2(y,x)函数的返回值范围(-PI,PI],表示与复数x+yi的幅角
        // segMsg.startOrientation范围为(-PI,PI]
        // segMsg.endOrientation范围为(PI,3PI]
        // 因为内部雷达旋转方向原因，所以atan2(..)函数前面需要加一个负号
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        // 下面这句话怀疑作者可能写错了，laserCloudIn->points.size() - 2应该是laserCloudIn->points.size() - 1
        segMsg.endOrientation   = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                                     laserCloudIn->points[laserCloudIn->points.size() - 2].x) + 2 * M_PI;
		// 开始和结束的角度差一般是多少？
		// 一个velodyne 雷达数据包转过的角度多大？
        // 雷达一般包含的是一圈的数据，所以角度差一般是2*PI，一个数据包转过360度

		// segMsg.endOrientation - segMsg.startOrientation范围为(0,4PI)
        // 如果角度差大于3Pi或小于Pi，说明角度差有问题，进行调整。
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
            segMsg.endOrientation -= 2 * M_PI;
        } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
		// segMsg.orientationDiff的范围为(PI,3PI),一圈大小为2PI，应该在2PI左右
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
    }

    // 将点云转换成矩阵
    void projectPointCloud(){
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize; 
        PointType thisPoint;

        cloudSize = laserCloudIn->points.size();

        for (size_t i = 0; i < cloudSize; ++i){

            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;

            // 计算竖直方向上的角度（雷达的第几线），这里是角度，不是弧度
            // verticalAngle = [-15.x,+15.x]
            verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
			
            // rowIdn计算出该点激光雷达是竖直方向上第几线的
			// 从下往上计数，-15度记为初始线，第0线，一共16线(N_SCAN=16)
            rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            // atan2(y,x)函数的返回值范围(-PI,PI],表示与复数x+yi的幅角
            // 以y轴正方向(左手边)为0点，计算该点所在角度
            // 顺时针为正
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

			// round函数进行四舍五入取整
            // 车头方向==x轴方向， 对应的horizonAngle=90
            // x轴正方向所在的线束列编号为 Horizon_SCAN/2
            // x轴负方向所在线束列编号为 0
            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            // 计算距离
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            // 填充分割矩阵
            rangeMat.at<float>(rowIdn, columnIdn) = range;

			// columnIdn:[0,H] (H:Horizon_SCAN)==>[0,1800]
            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            // 计算点在一维数组的索引
            index = columnIdn  + rowIdn * Horizon_SCAN;
            // 保存点
            fullCloud->points[index] = thisPoint;

            fullInfoCloud->points[index].intensity = range;
        }
    }

    // 取16线雷达的第0~6根线上的点
    // 计算相邻线束之间同一方向上同时达到的两个点，得到仰角
    // 仰角<10，认为是地面点
    void groundRemoval(){
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;

        // 遍历一圈的点数
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            // groundScanInd 是在 utility.h 文件中声明的线数，groundScanInd=7
            // 取第 0～6根线
            for (size_t i = 0; i < groundScanInd; ++i){

                // 当前这一圈的第j点
                lowerInd = j + ( i )*Horizon_SCAN;
                // 向上一圈的第j点
                upperInd = j + (i+1)*Horizon_SCAN;

                // 初始化的时候用fullCloud->points[]使用nanPoint填充，因此初始化时 .intensity = -1
                // 然后经过了分割，分割之后， 正常的点 .intensity !=-1
                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1){
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }

                // 由上下两线之间点的XYZ位置得到俯仰角
				// 如果俯仰角在10度以内，则判定(i,j)为地面点,groundMat[i][j]=1
				// 否则，则不是地面点，进行后续操作
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;
                // 计算仰角
                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                if (abs(angle - sensorMountAngle) <= 10){
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }

		// 找到所有点中的地面点或者距离为FLT_MAX(rangeMat的初始值)的点，并将他们标记为-1
		// rangeMat[i][j]==FLT_MAX，代表的含义是什么？ 无效点
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                // 如果被标记为地面点 或者 是分割的时候没有值的点 ， 使用标记矩阵记录
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }

        int ground_point_count=0;
		// 如果有节点订阅groundCloud，那么就需要把地面点发布出来
		// 具体实现过程：把点放到groundCloud队列中去
        if (pubGroundCloud.getNumSubscribers() != 0){
            for (size_t i = 0; i <= groundScanInd; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (groundMat.at<int8_t>(i,j) == 1){
                        ground_point_count++;
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    }
                }
            }
        }
        std::cout<<"提取地面点数:"<<ground_point_count<<std::endl;
    }

    void cloudSegmentation(){
        // 对没有label的点进行聚类分类
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                // 如果labelMat[i][j]=0,表示没有对该点进行过分类
				// 需要对该点进行聚类
                if (labelMat.at<int>(i,j) == 0)
                    labelComponents(i, j);

        int sizeOfSegCloud = 0;
        // 按线束遍历
        for (size_t i = 0; i < N_SCAN; ++i) {
			
			// segMsg.startRingIndex[i]
			// segMsg.endRingIndex[i]
			// 表示第i线的点云起始序列和终止序列
			// 以开始线后的第6线为开始，以结束线前的第6线为结束
            // segMsg.startRingIndex[0] = 4;
            // 因为要取前后5个点来计算粗糙度，因此这里有个 +5?
            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;

            // 按列遍历
            for (size_t j = 0; j < Horizon_SCAN; ++j) {
				// 找到可用的特征点或者地面点(不选择labelMat[i][j]=0的点)
                if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
					// labelMat数值为999999表示这个点是因为聚类数量不够30而被舍弃的点
					// 需要舍弃的点直接continue跳过本次循环，
					// 当列数为5的倍数，并且行数较大，可以认为非地面点的，将它保存进异常点云(界外点云)中
					// 然后再跳过本次循环
                    if (labelMat.at<int>(i,j) == 999999){
                        if (i > groundScanInd && j % 5 == 0){
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                            continue;
                        }else{
                            continue;
                        }
                    }
                    
					// 如果是地面点,对于列数不为5的倍数的，直接跳过不处理
                    // 地面点，并且列索引在(5,1800-5)范围内的，也跳过
                    // 这里的操作目的是：从地面点中选出某几个点，用来参与后面的粗糙度计算?
                    if (groundMat.at<int8_t>(i,j) == 1){
                        if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                            continue;
                    }
					// 上面多个if语句已经去掉了不符合条件的点，这部分直接进行信息的拷贝和保存操作
					// 保存完毕后sizeOfSegCloud递增
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1); //是否地面点标志
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;                                    //该点在矩阵的列号
                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);              //距离
                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);                   //保存该点到segmentedCloud
                    ++sizeOfSegCloud;
                }
            }

            // 以该线束最后一个点的索引-5作为结束
            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }

        printf("当前帧提取的点数: %d\n",segmentedCloud->size());

		// 如果有节点订阅SegmentedCloudPure,
		// 那么把点云数据保存到segmentedCloudPure中去
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            // 遍历
            for (size_t i = 0; i < N_SCAN; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    // 需要选择 不是地面点(labelMat[i][j]!=-1)和 没被舍弃的点
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                        // segmentedCloudPure: pcl点云类型
                        // 记录点
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        // 使用强度位来记录该点所属label
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
                    }
                }
            }
        }
    }

    // 使用BFS扩展搜索，进行聚类，给labelMat矩阵进行赋值
    void labelComponents(int row, int col){
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY; 
        bool lineCountFlag[N_SCAN] = {false};

        // 查询点行列
        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;
        
        // 标准的BFS
        // BFS的作用是以(row，col)为中心向外面扩散，
        // 判断(row,col)是否是这个平面中一点
        while(queueSize > 0){
            // 取队列起始
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            // 给当前点添加一个label
			// labelCount的初始值为1，后面会递增
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;

            // neighborIterator=[[-1,0];[0,1];[0,-1];[1,0]]
			// 遍历点[fromIndX,fromIndY]边上的四个邻点
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){

                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;

                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;

                // 是个环状的图片，左右连通
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;

				// 如果点[thisIndX,thisIndY]已经标记过
				// labelMat中，-1代表无效点，0代表未进行标记过，其余为其他的标记
				// 如果当前的邻点已经标记过，则跳过该点。
				// 如果labelMat已经标记为正整数，则已经聚类完成，不需要再次对该点聚类
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                // 这里主要是区分两个点，d1表示远的点，d2表示近的点
                // d1: 取邻点与中心点 最远距离的那个
                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                // d2: 取邻点与中心点 最近距离的那个(相对于激光雷达原点)
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));

                // 根据水平和垂直方向，选择系数
				// alpha代表角度分辨率，
				// X方向上角度分辨率是segmentAlphaX(rad)
				// Y方向上角度分辨率是segmentAlphaY(rad)
                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                else
                    alpha = segmentAlphaY;

				// 通过下面的公式计算这两点之间是否有平面特征
				// atan2(y,x)的值越大，d1，d2之间的差距越小,越平坦
                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                if (angle > segmentTheta){
					// segmentTheta=1.0472<==>60度
					// 如果算出角度大于60度，则假设这是个平面

                    // 把这个邻近点加入到搜索队列
                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    // 给邻近点打标签
                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;     //?

                    // 只要是大于阈值的点，都保存到allPushedIndX[] ,allPushedIndY[]
                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }


        bool feasibleSegment = false;

		// 如果聚类超过30个点，直接标记为一个可用聚类，labelCount需要递增
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        else if (allPushedIndSize >= segmentValidPointNum){
			// 如果聚类点数小于30大于等于5，统计竖直方向上的聚类点数
            int lineCount = 0;
            // 在不同线束上进行统计,以表示竖直方向
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;

			// 竖直方向上超过3个也将它标记为有效聚类
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;            
        }

        if (feasibleSegment == true){
            ++labelCount;
        }else{
            for (size_t i = 0; i < allPushedIndSize; ++i){
                // 如果到了这里，表示 allPushedIndSize<30，同时，竖直方向也没有关联聚类，因此，舍弃
                // 标记为999999的是需要舍弃的聚类的点
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

    // 发布各类点云内容
    void publishCloud(){
        // segMsg: 当前帧扫描的一些信息，不包含点云
    	// 发布cloud_msgs::cloud_info消息
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);

        sensor_msgs::PointCloud2 laserCloudTemp;

		// pubOutlierCloud发布界外点云
        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);

        // pubSegmentedCloud发布下一步用到的点云
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);

        // 该次扫描的全部点
        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }

        // 地面点
        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }

        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }

        if (pubFullInfoCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }
    }
};




int main(int argc, char** argv){

    ros::init(argc, argv, "lego_loam");
    
    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
