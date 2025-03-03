clc
clear

%% Change Folder Name Here to select datasets
list = dir('D:\SW\数据\data-All-按疾病分类');
menupath = list.folder;
listname = {list.name};
listname = {listname{3:end}};
for j = 21 %1 : size(listname,2)
    datalistpath = fullfile(menupath, listname{j}, '*.mat');
    datalist = dir(datalistpath);
    path = datalist.folder;
    filename = {datalist.name};
    save_path = fullfile('D:\SW\数据\denoise-All-按疾病分类', listname{j});
    PCA_path = fullfile('D:\SW\数据\PCA-Trichannel-按疾病分类', listname{j});
    STFT_path = fullfile('D:\SW\数据\STFT-Trichannel-按疾病分类', listname{j});
    if ~exist(save_path, 'dir')
        mkdir(save_path);
    end
    if ~exist(PCA_path, 'dir')
        mkdir(PCA_path);
    end
    if ~exist(STFT_path, 'dir')
        mkdir(STFT_path);
    end
    %% Loop
    for k = 1 : size(datalist,1)
    %% Read Target imformation from the data
    
    filepath = strcat(path,'\',filename(1, k));
    load(filepath{1});
    data = AD_File; %P.data(28 :30, :);   
    index = size(data,1);
    Target = ['寸', '关', '尺'];

    %% Preprocess
    fprintf('正在处理%s : %s 编号%d\n',listname{j}, cell2mat(filename(1, k)), k);
    P_denoise = [];
    % Preprocess
        for i = 1 : index
            [tt_cycle{i},pp,fp,B]=Ad_diff(data(i, :)');
            data_cycle{i}=mean(outlier_check(tt_cycle{i},fp,64,1),1);
            tt_cycle{i}=outlier_check(tt_cycle{i},fp,64,1);
            if size(tt_cycle{i}, 1) > 10
                P_denoise(i, :) = B(200:end-200);
                PCA(i, :)=feaPCA(tt_cycle{i});
                STFT(i, :)=feaSTFT(data_cycle{i});
            else
                P_denoise=[];
                break
            end       
        end
        
        if ~isempty(P_denoise)
            fprintf('%s : %s数据有效\n',listname{j}, cell2mat(filename(1, k)));
            file_path = cell2mat(strcat(strcat(save_path,'\', filename(1, k))));
            pca_file_path = cell2mat(strcat(strcat(PCA_path,'\', filename(1, k))));
            stft_file_path = cell2mat(strcat(strcat(STFT_path,'\', filename(1, k))));
            save(file_path, 'P_denoise');
            save(pca_file_path, 'PCA');
            save(stft_file_path, 'STFT');
        else
            fprintf('%s : %s数据无效\n',listname{j}, cell2mat(filename(1, k)));
        end
    end
end