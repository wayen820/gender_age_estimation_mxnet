# gender_age_estimation_mxnet
gender and age estimation
这是一个mxnet版本的SSR的年龄和性别识别，作者原版链接在这里 https://github.com/shamangary/SSR-Net

训练方法
1）下载数据，imdb-wiki数据（人脸），下载链接https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
使用  src/data/目录下process_data_wiki_imdb.py文件来打包数据到mxnet里面.rec格式

2）下载亚洲年龄Megaage-Asian数据，下载链接https://github.com/b02901145/SSR-Net_megaage-asian
使用  src/data/目录下process_data_mege_asia.py来打包亚洲人数据

3）训练年龄模型，用train_ssr_adam.sh首先在imdb上面训练，再用train_ssr_adam_mege.sh来在亚洲人上finetune，得到年龄模型，mae在亚洲人脸验证集最终达到4.8左右

4）训练性别模型，用train_ssr_adam_gender在imdb上面训练，得到性别模型，mae在验证集上能达到1.1-1.2

测试及ncnn上部署方法
1）使用deploy中model_slim_age.py和model_sim_gender.py来裁剪模型，ncnn中不支持range操作，裁剪将range操作替换为输入
2）使用test.py来从摄像头测试


