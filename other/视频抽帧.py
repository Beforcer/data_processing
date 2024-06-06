# encoding:utf-8

'''
读取文件列表，将视频文件单独拿出来，隔帧抽图并保存
'''
import cv2
import os
import glob
if __name__ == '__main__':
    # 获取目标路径下所有mp4文件
    mp4s = glob.glob(r'D:\MyData\data_flydetection\shenyang\定位图像\video1\shenyang.mp4')
    for one_mp4 in mp4s:
        print(one_mp4)
        # 要提取视频的文件名，隐藏后缀
        video_path = one_mp4
        sourceFileName = os.path.basename(video_path).split('.')[0]

        # 在这里把后缀接上
        # video_path = os.path.join("", "", sourceFileName + '.mp4')
        # video_path = os.path.join("", "", sourceFileName)
        times = 0
        namenumber = 0
        loopNum = 0
        # 提取视频的频率，每1帧提取一个
        frameFrequency = 5
        # 输出图片到当前目录vedio文件夹下
        # outPutDirName = 'VedioPicture/' + sourceFileName + '/'
        # outPutDirName = 'E:/data/picture/' + sourceFileName + '/'
        outPutDirName = r'D:\MyData\data_flydetection\shenyang\定位图像\video1/' + sourceFileName + '/'
        if not os.path.exists(outPutDirName):
            # 如果文件目录不存在则创建目录
            os.makedirs(outPutDirName)
        camera = cv2.VideoCapture(video_path)
        while True:
            times += 1

            loopNum += 1
            res, image = camera.read()
            if not res:
                print('not res , not image')
                break
            if times % frameFrequency == 0:
                namenumber += 1
                if (loopNum < 10):
                    namenumber2str = sourceFileName + '00000' + str(namenumber)
                elif (loopNum < 100):
                    namenumber2str = sourceFileName + '0000' + str(namenumber)
                elif (loopNum < 1000):
                    namenumber2str = sourceFileName + '000' + str(namenumber)
                elif (loopNum < 10000):
                    namenumber2str = sourceFileName + '00' + str(namenumber)
                elif (loopNum < 100000):
                    namenumber2str = sourceFileName + '0' + str(namenumber)
                else:
                    namenumber2str = str(namenumber)
                # cv2.imwrite(outPutDirName + namenumber2str + '.jpg', image)  # sourceFileName + '_' +
                cv2.imencode('.jpg', image)[1].tofile(outPutDirName + namenumber2str + '.jpg')
                print (outPutDirName + namenumber2str + '.jpg')
        print (one_mp4 + '图片提取结束')
        camera.release()
