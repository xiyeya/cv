

clear;
% ------------------------------------------------------------------------------
% Attenuation coefficient (in inverse meters).
% In case of fog, the value of the attenuation coefficient
% should be set to 0.003 or higher.
beta = 0.03;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ;
% ------------------------------------------------------------------------------
Path = 'D:\ILSVRC\Data\VID\snippets\test\';                   % 设置数据存放的文件夹路径
File = dir(fullfile(Path,'*.mp4'));  % 显示文件夹下所有符合后缀名为.txt文件的完整信息
FileNames = {File.name}';            % 提取符合后缀名为.txt的所有文件的文件名，转换为n行1列
Length_Names = size(FileNames,1);    % 获取所提取数据文件的个数
for k = 1 : Length_Names
    name = FileNames(k);
    videoname = name{1,1};
    videopath = strcat(Path, videoname);
    % ------------------------------------------------------------------------------
    % preserve all frames in a video
    obj=VideoReader(videopath);
    num=obj.NumberOfFrames;
    for i = 1:num  
        frame=read(obj,i);
        out_path=strcat('..\data\demo\img\',num2str(i));  %帧图输出路径
        out_path=strcat(out_path,'.jpg');
        imwrite(frame,out_path)
    end 

    % estimate depth.
    py.demo.main()

    for i= 1:num
      image = int2str(i);
      CreateFog(image, beta);
    end

    WriterObj = VideoWriter(strcat('D:\SynFogVID\Data\snippets\test\', videoname, num2str(beta,'%.2f')), 'MPEG-4');
    % ('master of shadow.avi', 'Uncompressed AVI');   %这里输出的路径是默认路径，合成的视频的格式是avi
         %avi格式的话光是影 流 之 主那个14秒的视频都要1个多G，过于高清，可以改为mp4，这样就合成的视频比较小
         %改为mp4格式只需将内容改为：VideoWriter('master of shadow.mp4', 'MPEG-4')
    open(WriterObj);
    for i=1:num  %帧图数量
      pic='..\output\demo\foggy_img\';  %前面边缘检测的图片的存储路径
      pic=strcat(pic,num2str(i));
      ppic=strcat(pic,'.png');
      frame=imread(ppic);  % 读取图像，放在变量frame中
      writeVideo(WriterObj,frame);  % 将frame放到变量WriterObj中
    end
    close(WriterObj);

    delete('..\output\demo\foggy_img\*.png');
    delete('..\output\demo\transmittance_map\*.png');
    delete('..\data\demo\img\*.jpg');
    delete('..\data\demo\depth_mat\*.mat');

    fprintf("Finsh create foggy video %s\n", videoname)
    
end





