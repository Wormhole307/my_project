#my_facenet_project
tensorflow+python+opencv+mtcnn+facenet 实现人脸识别考勤系统

Abstract:本文记录了在学习深度学习过程中，使用tensorflow+python+opencv+mtcnn+facenet，实现人脸识别考勤系统，开发环境为windows 10，
	实现利用电脑摄像头，对考勤人员进行实时人脸识别考勤，该代码只是用于学习，还存在很多不足，会持续优化


### 目录结构


|-----20170512-110547 				>	 文件夹是facent的预训练模型
|-----align 									>	 第三方MTCNN模型中pnet, rnet, onet网络结构
|-----img 										>	 录入人脸图片保存目录
|-----logs 										>	 录入人脸图片和姓名关联信息保存目录
|-----AttendanceSystem.py 		>	 开始考勤系统
|-----facenet.py							>	 第三方facenet模型中的相关方法
|-----InputSystem.py					>	 人脸信息录入
|-----my_attendance_table.sql		>	 mysql DDL语句
|-----my_facenet_table.sql		>	     mysql DDL语句
|-----UI_Main.py							>	 人脸信息录入UI界面


### 运行环境和运行说明

1. 使用Anaconda3配置tensorflow环境（因为本项目就是基于tensorflow框架的），是cpu 1.7.0 版本，python版本是3.6

2. 编辑器用的是pycharm

3. 依赖库，比如numpy scipy PIL mysql PyQt5 tensorflow 等

4. 本项目里的文件可以直接运行，运行顺序是

   1. 运行InputSystem.py,先输入采集人脸图像对应的姓名，再打开摄像头，这个时候可以点击拍照，图片自动保存在img目录下，
   同一个人脸可以点击多次拍照，图片越多，识别度越高，本人测试时不同角度的照片都有，共计5张，也建议每个人脸拍5找图片。
   同时会将人脸图片和人脸图像对应的姓名之间的关联关系保存在mysql数据库或者是logs目录下的的data.txt文件中。图片拍照后关闭即可
   
   2. 运行AttendanceSystem.py，会自动打开电脑摄像头，将头像对准显示框，这个时候就开始人脸识别，如果之前进行了人脸录入，就会显示考勤成功

   3. 启动的时候会比较慢，取决于电脑性能
	

### 阅读代码和重构项目建议

这个项目还有不足之处，包括启动慢，人脸识别中存在卡顿，没有进行活体检测等，这些都需要进行优化

代码思路

1. 使用mtcnn截取视频帧中人像的人脸，并拉伸为固定大小（这里为160*160,由使用的facenet网络所受限）
2. 将上一步骤得到的多张人脸图片输入facenet，得到embedding层的输出，一个人像对应一个1*128数据，并保存在内存中，然后将摄像头拍摄的每一帧有人像的图像
	 进行进行第一步处理，再进行第二部处理，生成128维的向量，然后分别计算之前保存在内存中的多个128维向量中的每一个向量的距离，如果之间的距离差距大于某个阀值，
	 就代表是不同的人脸，如果差距小于某个阀值，代表是同一个人脸


注意事项:

1. opencv在显示时中文乱码，opencv支持的中文编码是GBK，并不是UTF-8，一定要注意，但是还是对中文长度有限制。
2. opencv在读取图片时是BGR通道，需要进行转换为RGB通道
3. opencv在图像中无法显示中文，不要使用opencv自带的绘图方法，使用PIL库中的Image,ImageDraw,ImageFont模块，进行显示。
