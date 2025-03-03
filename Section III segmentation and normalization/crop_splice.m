clc
clear

%% Change Folder Name Here to select datasets
list = dir('D:\SW\数据\STFT-Trichannel-按疾病分类');
menupath = list.folder;
listname = {list.name};
listname = {listname{3:end}};
for j = 1 : size(listname,2)
    datalistpath = fullfile(menupath, listname{j}, '*.mat');
    datalist = dir(datalistpath);
    path = datalist.folder;
    filename = {datalist.name};
    save_path_crop = fullfile('D:\SW\数据\crop-Trichannel-按疾病分类', listname{j});
    save_path_splice = fullfile('D:\SW\数据\STFT-Tri2one-按疾病分类', listname{j});
    if ~exist(save_path_crop, 'dir')
        mkdir(save_path_crop);
    end
    if ~exist(save_path_splice, 'dir')
        mkdir(save_path_splice);
    end
    %% Loop
    for k = 1 : size(datalist,1)
        %% Read Target imformation from the data

        filepath = strcat(path,'\',filename(1, k));
        load(filepath{1});
        data = STFT;   
        index = size(data,1);
        ResMenu = [];
%         if size(data, 2) >= 10000
            %% Preprocess
            fprintf('正在处理%s : %s\n',listname{j}, cell2mat(filename(1, k)));
      
%             crop = data(:, round(size(data, 2)/2) - 4999 : round(size(data, 2)/2) + 5000);
%             file_path_crop = cell2mat(strcat(strcat(save_path_crop,'\', filename(1, k))));
%             save(file_path_crop, 'crop');
            splice = [data(1, :), data(2, :), data(3, :)];
            file_path_splice = cell2mat(strcat(strcat(save_path_splice,'\', filename(1, k))));
            save(file_path_splice, 'splice');
%         else
%            ResMenu = [ResMenu; strcat(listname{j}, filename(1, k))];
%         end
%         save(save_path_crop, 'ResMenu')
    end       
end
