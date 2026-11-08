function [feat, label, dataset_name, group] = Load_UCI_Data(DataSet_Index)

    if nargin == 0
        clc
        close all
        clear all
        DataSet_Index = 34;
    end

    dataset_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'Dataset');
    addpath(dataset_dir);

    dataset_name = 'Unnamed' ;
    group = [];   % Group ID per sample (patient/molecule) for group-level splitting; empty for datasets without group structure

    % More datasets can be found from the UCI Machine Learning Repository
    % https://archive.ics.uci.edu/ml/datasets.php
    % Feel free to expand the datasets by increasing the case number in the switch-case
    switch DataSet_Index
    
        case 1
            % Breast Cancer Wisconsin (Original) Data Set
            % https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
            disp('Loading DS1 - Breast Cancer Wisconsin (Original) Data Set')
            fileID = fopen(fullfile(dataset_dir, 'breast-cancer-wisconsin.data'));
            %fmt = repmat('%f', [569, 11]);
            format = repmat('%f', [1, 11]);
            C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(1:end,2:end-1);
            label1 = C{1,1}(:,end);
            index_2 = find(label1 == 2);
            index_4 = find(label1 == 4);
            label(index_2,1) = 0;
            label(index_4,1) = 1;
            x=1;

            dataset_name = '1 Breast Cancer Original' ;
    
        case 2
            % Breast Cancer Wisconsin (Diagnostic) Data Set
            % https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
            disp('Loading DS 2 - Breast Cancer Wisconsin (Diagnostic) Data Set')
            fileID = fopen(fullfile(dataset_dir, 'wdbc.data'));
            format = ['%f%s' repmat('%f', [1, 30])];
            C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,3};
            label1 = cell2mat(C{1,2});
            index_M = find(label1 == 'M');
            index_B = find(label1 == 'B');
            label(index_M,1) = 0;
            label(index_B,1) = 1;
            
            dataset_name = '2 Breast Cancer Diagnostic' ; 

        case 3
            % Dermatology Data Set
            % https://archive.ics.uci.edu/ml/datasets/Dermatology
            disp('Loading DS 3 - Dermatology Data Set')
            fileID = fopen(fullfile(dataset_dir, 'dermatology.data'));
            format = repmat('%f', [1, 35]);
            C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(:, 1:end-1);
            label = C{1,1}(:, end);

            dataset_name = '3 Dermatology' ; 
    
        case 4
            % Glass Identification Data Set
            % https://archive.ics.uci.edu/ml/datasets/Glass+Identification
            disp('Loading DS 4 - BGlass Identification Data Set')
            fileID = fopen(fullfile(dataset_dir, 'glass.data'));
            format = repmat('%f', [1, 11]);
            C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(:, 2:end-1); %exclude ID at column 1
            label = C{1,1}(:, end);
            dataset_name = '4 Glass Identification' ; 
        case 5
            % Lymphography Data Set
            % https://archive.ics.uci.edu/ml/datasets/Lymphography
            disp('Loading DS 5 - Lymphography Data Set')
            fileID = fopen(fullfile(dataset_dir, 'lymphography.data'));
            format = repmat('%f', [1, 19]);
            C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(:, 1:end-1);
            label = C{1,1}(:, end);
            dataset_name = '5 Lymphography' ; 
 
        case 6
            % Lung Cancer Data Set
            % https://archive.ics.uci.edu/ml/datasets/Lung+Cancer
            disp('Loading DS 6 - Lung Cancer Data Set')
            fileID = fopen(fullfile(dataset_dir, 'lung-cancer.data'));
            format = repmat('%f', [1, 57]);
            C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);

            % data = C{1};  % assume C{1} is a matrix
            % writematrix(data, fullfile(dataset_dir, 'mydata.csv'));  % save as CSV

            fclose(fileID);
            feat = C{1,1}(:, 2:end);
            label = C{1,1}(:, 1); % Attribute 1 is class label
            
            dataset_name = '6 Lung Cancer' ; 

        case 7
            % Multiple Features Data Set
            % https://archive.ics.uci.edu/ml/datasets/Multiple+Features
            disp('Loading DS 7 - Multiple Features Data Set')
            load(fullfile(dataset_dir, 'mfeat-fac')) %216 features
            load(fullfile(dataset_dir, 'mfeat-fou')) %76 features
            load(fullfile(dataset_dir, 'mfeat-kar')) %64 features
            load(fullfile(dataset_dir, 'mfeat-mor')) %6 features
            load(fullfile(dataset_dir, 'mfeat-pix')) %240 features
            load(fullfile(dataset_dir, 'mfeat-zer')) %47 features
            feat = [mfeat_fac,mfeat_fou,mfeat_kar,...
                mfeat_mor,mfeat_pix,mfeat_zer];
            [data_size, ~] = size(feat);
            label = zeros(data_size,1);
            label(1:200,1) = 0;
            label(201:400,1) = 1;
            label(401:600,1) = 2;
            label(601:800,1) = 3;
            label(801:1000,1) = 4;
            label(1001:1200,1) = 5;
            label(1201:1400,1) = 6;
            label(1401:1600,1) = 7;
            label(1601:1800,1) = 8;
            label(1801:2000,1) = 9;
            dataset_name = '7 Multiple Features' ; 

        case 8
            % Statlog (Heart) Data Set
            % https://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29
            disp('Loading DS 8 - Statlog (Heart) Data Set')
            fileID = fopen(fullfile(dataset_dir, 'heart.dat'));
            format = repmat('%f', [1, 14]);
            C = textscan(fileID, format, 'Delimiter',' ', 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(:, 1:end-1);
            label = C{1,1}(:, end);
            dataset_name = '8 Statlog (Heart)' ; 
        case 9
            % Ionosphere Data Set
            % https://archive.ics.uci.edu/ml/datasets/ionosphere
            % Built in dataset in Matlab
            disp('Loading DS 9 - Ionosphere Data Set')
            load ionosphere.mat;
            feat = X;
            [Total_Data, ~] = size(feat);
            label = zeros(Total_Data,1);
            for i=1:Total_Data
                if isequal(Y{i},'g')
                    label(i)= 1;
                elseif isequal(Y{i},'b')
                    label(i)= 0;
                end
            end
            dataset_name = '9 Ionosphere' ; 
        case 10
            % Iris Data Set
            % https://archive.ics.uci.edu/ml/datasets/iris
            % Built in dataset in Matlab
            disp('Loading DS 10 - Iris Data Set')
            load fisheriris.mat
            feat = meas;
            [Total_Data, ~] = size(feat);
            label = zeros(Total_Data,1);
            for i = 1:Total_Data
                if isequal(species{i},'setosa')
                    label(i)= 1;
                elseif isequal(species{i},'versicolor')
                    label(i)= 2;
                elseif isequal(species{i},'virginica')
                    label(i)= 3;
                end
            end
            dataset_name = '10 Iris' ; 
        case 11
            % Arrhythmia Data Set
            % https://archive.ics.uci.edu/ml/datasets/arrhythmia
            % Built in dataset in Matlab
            disp('Loading DS 11- Arrhythmia Data Set')
            load(fullfile(dataset_dir, 'arrhythmia.mat'))
            feat = X;
            label = Y;
            dataset_name = '11 Arrhythmia' ;  
        case 12
            % Ovarian Data Set
            % Built in dataset in Matlab
            disp('Loading DS 12 - Ovarian Data Set')
            load(fullfile(dataset_dir, 'ovariancancer.mat'))
            feat = obs;
            [Total_Data, ~] = size(feat);
            label = zeros(Total_Data,1);
            for i = 1:Total_Data
                if isequal(grp{i},'Cancer')
                    label(i)= 1;
                elseif isequal(grp{i},'Normal')
                    label(i)= 0;
                end
            end
            dataset_name = '12 Ovarian' ;  
        case 13
            % Echocardiogram Data Set
            % https://archive.ics.uci.edu/ml/datasets/Echocardiogram
            disp('Loading DS 13 - Echocardiogram Data Set')
            fileID = fopen(fullfile(dataset_dir, 'echocardiogram.data'));
            format = [repmat('%s', [1, 13])]; % 13 Row in the table
            C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
            fclose(fileID);
            c1 = C{1,1}(1:end,[1:7,9,13]); % Remove meaningless attribute
            c1(51,:) = [];
            j=1;
            for i = 1:132 % Remove unknowmn value
                test = find(strcmp(c1(i,:), '?'));
                if numel(test) == 0
                    c2(j,:)=c1(i,:);
                    j=j+1;
                else
                end
            end
            for i = 1:61 % Convert character array to numeric array
                for j = 1:8
                    feat(i,j) = str2num(c2{i,j});
                end
                label(i,:) = str2num(c2{i,end});
            end
            dataset_name = '13 Echocardiogram' ;  
        case 14
            % Haberman's Survival Data Set
            % https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival
            disp('Loading DS 14 - Haberman Survival Data Set')
            fileID = fopen(fullfile(dataset_dir, 'haberman.data'));
            format = [repmat('%f', [1, 4])];
            C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(1:end,1:end-1);
            label = C{1,1}(1:end,end);
            dataset_name = '14 Habermans Survival' ;  
        case 15
            % Liver Disorders Data Set
            % https://archive.ics.uci.edu/ml/datasets/Liver+Disorders
            disp('Loading DS 15 - Liver Disorders Data Set')
            fileID = fopen(fullfile(dataset_dir, 'bupa.data'));
            format = [repmat('%f', [1, 7])];
            C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(1:end, 1:end-1);
            label = C{1,1}(1:end, end);
            dataset_name = '15 Liver Disorders' ;
        case 16
            % Diabetes Data Set
            % https://www.kaggle.com/uciml/pima-indians-diabetes-database
            % https://github.com/jbrownlee/Datasets/blob/master/pima-indians-diabetes.data.csv
            disp('Loading DS 16 - Pima Indians Diabetes Data Set')
            C = readmatrix(fullfile(dataset_dir, 'pima-indians-diabetes.data.csv'));
            feat = C(1:end, 1:end-1);
            label = C(1:end, end);
            dataset_name = '16 Diabetes' ;
        case 17
            % MONK 1 Problem Data Set
            % https://archive.ics.uci.edu/ml/datasets/MONK's+Problems
            disp('Loading DS 17 - MONK 1 Problem Data Set')
            fileID = fopen(fullfile(dataset_dir, 'monks-1.train.txt'));
            format = [repmat('%f', [1, 7]) '%s'];
            C = textscan(fileID, format, 'EmptyValue', -Inf, 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(1:end, 2:end);
            label = C{1,1}(1:end, 1);
            dataset_name = '17 MONK 1 Problem' ;
        case 18
            % Connectionist Bench (Sonar, Mines vs. Rocks) Data Set
            % https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29
            disp('Loading DS 18 - Connectionist Bench (Sonar, Mines vs. Rocks) Data Set')
            fileID = fopen(fullfile(dataset_dir, 'sonar.all-data'));
            format = [repmat('%f', [1, 60]) '%s'];
            C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1};
            label1 = C{1,2};
            [Total_Data, ~] = size(feat);
            label = zeros(Total_Data,1);
            for i=1:Total_Data
                if label1{i} == 'M'        % Mines
                    label(i,:) = 1;
                else % if label1{i} == 'R' % Rock
                    label(i,:) = 0;
                end
            end
            dataset_name = '18 Connectionist Bench' ;
        case 19
            % Parkinsons Data Set
            % https://archive.ics.uci.edu/ml/datasets/Parkinsons
            disp('Loading DS 19 - Parkinsons Data Set')
            fileID = fopen(fullfile(dataset_dir, 'parkinsons.data'));
            format = ['%s' repmat('%f', [1, 23])];
            C = textscan(fileID, format, 'Delimiter',',', 'headerlines', 1, 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,2}(1:end, 1:end ~= 17);
            label = C{1,2}(1:end, 17);
            dataset_name = '19 Parkinsons' ;
        case 20
            % Waveform Database Generator (Version 1) Data Set
            % https://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+%28Version+1%29
            disp('Loading DS 20 - Waveform Database Generator (Version 1) Data Set')
            fileID = fopen(fullfile(dataset_dir, 'waveform.data'));
            format = [repmat('%f', [1, 22])];
            C = textscan(fileID, format, 'Delimiter',',', 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(1:end, 1:end-1);
            label = C{1,1}(1:end, end);
            dataset_name = '20 Waveform' ;
        case 21
            % Yeast Data Set
            % https://archive.ics.uci.edu/ml/datasets/Yeast
            disp('Loading DS 21 - Yeast Data Set')
            fileID = fopen(fullfile(dataset_dir, 'yeast.data'));
            format = [ '%s' repmat('%f', [1, 8]) '%s'];
            C = textscan(fileID, format, 'EmptyValue', -Inf, 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,2};
            [Total_Data, ~] = size(feat);
            label = zeros(Total_Data,1);
            for i = 1:Total_Data
                if isequal(C{1,3}{i},'CYT')
                    label(i)= 1;
                elseif isequal(C{1,3}{i},'ERL')
                    label(i)= 2;
                elseif isequal(C{1,3}{i},'EXC')
                    label(i)= 3;
                elseif isequal(C{1,3}{i},'ME1')
                    label(i)= 4;
                elseif isequal(C{1,3}{i},'ME2')
                    label(i)= 5;
                elseif isequal(C{1,3}{i},'ME3')
                    label(i)= 6;
                elseif isequal(C{1,3}{i},'MIT')
                    label(i)= 7;
                elseif isequal(C{1,3}{i},'NUC')
                    label(i)= 8;
                elseif isequal(C{1,3}{i},'POX')
                    label(i)= 9;
                elseif isequal(C{1,3}{i},'VAC')
                    label(i)= 10;
                end
            end
            dataset_name = '21 Yeast' ;
        case 22
            % Blood Transfusion Service Center Data Set
            % https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
            disp('Loading DS 22 - Blood Transfusion Service Center Data Set')
            fileID = fopen(fullfile(dataset_dir, 'transfusion.data'));
            format = [repmat('%f', [1, 5])];
            C = textscan(fileID, format, 'Delimiter',',', 'headerlines', 1, 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(1:end, 1:end-1);
            label = C{1,1}(1:end, 5);
            dataset_name = '22 Blood Transfusion Service Center' ;
        case 23
            % Wine Data Set
            % https://archive.ics.uci.edu/ml/datasets/Wine
            disp('Loading DS 23 - Wine Data Set')
            fileID = fopen(fullfile(dataset_dir, 'wine.data'));
            format = [repmat('%f', [1, 14])];
            C = textscan(fileID, format, 'Delimiter', ',' , 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(1:end, 2:end);
            label = C{1,1}(1:end, 1);
            dataset_name = '23 Wine' ;
        case 24
            % Maternal Health Risk Data Set
            % https://archive.ics.uci.edu/ml/datasets/Maternal+Health+Risk+Data+Set
            disp('Loading DS 24 - Maternal Health Risk Data Set Data Set')
            fileID = fopen(fullfile(dataset_dir, 'Maternal Health Risk Data Set.csv'));
            format = [repmat('%f', [1, 6]) '%s'];
            C = textscan(fileID, format, 'Delimiter', ',' , 'headerlines', 1, 'CollectOutput', 1);
            fclose(fileID);
            a=unique(C{1,2});
            feat = C{1,1};
            [Total_Data, ~] = size(feat);
            label = zeros(Total_Data,1);
            for i = 1:Total_Data
                if isequal(C{1,2}{i},'low risk')
                    label(i)= 1;
                elseif isequal(C{1,2}{i},'mid risk')
                    label(i)= 2;
                elseif isequal(C{1,2}{i},'high risk')
                    label(i)= 3;
                end
            end
            dataset_name = '24 Maternal Health Risk' ;
        case 25
            % Zoo Data Set
            % https://archive.ics.uci.edu/ml/datasets/Zoo
            disp('Loading DS 25 - Zoo Data Set')
            fileID = fopen(fullfile(dataset_dir, 'zoo.data'));
            format = ['%s' repmat('%f', [1, 17])];
            C = textscan(fileID, format, 'Delimiter', ',' , 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,2}(1:end, 1:end-1);
            label = C{1,2}(1:end, end);
            dataset_name = '25 Zoo' ;
        case 26
            % ILPD (Indian Liver Patient Dataset) Data Set
            % https://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29
            disp('Loading DS 26 - ILPD (Indian Liver Patient Dataset) Data Set')
            fileID = fopen(fullfile(dataset_dir, 'Indian Liver Patient Dataset (ILPD).csv'));
            format = ['%f%s' repmat('%f', [1, 9])];
            C = textscan(fileID, format, 'Delimiter', ',' , 'CollectOutput', 1);
            fclose(fileID);
            feat = [C{1,1} C{1,3}(1:end, 1:end-1)];
            label = C{1,3}(1:end, end);
            dataset_name = '26 ILPD' ;
        case 27
            % Semeion Handwritten Digit Data Set
            % https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit
            disp('Loading DS 27 - Semeion Handwritten Digit Data Set')
            fileID = fopen(fullfile(dataset_dir, 'semeion.data'));
            format = [repmat('%f', [1, 266])];
            C = textscan(fileID, format, 'EmptyValue', -Inf, 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(1:end, 1:end-10);
            [~,label]=max(C{1,1}(1:end, 257:end),[],2); % Find '1' column value in every row
            dataset_name = '27 Semeion Handwritten Digit' ;
         case 28
            % Letter Recognition Data Set
            % https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
            disp('Loading DS 28 - Letter Recognition Data Set')
            fileID = fopen(fullfile(dataset_dir, 'letter-recognition.data'));
            format = ['%s' repmat('%f', [1, 16])];
            C = textscan(fileID, format, 'Delimiter', ',' , 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,2};
            [Total_Data, ~] = size(feat);
            label = zeros(Total_Data,1);
            for i = 1:Total_Data
                label(i) = double(C{1,1}{i})-64;
            end
            dataset_name = '28 Letter Recognition' ;
         case 29
            % Balance Scale Data Set
            % https://archive.ics.uci.edu/ml/datasets/Balance+Scale
            disp('Loading DS 29 - Balance Scale Data Set')
            fileID = fopen(fullfile(dataset_dir, 'balance-scale.data'));
            format = ['%s' repmat('%f', [1, 4])];
            C = textscan(fileID, format, 'Delimiter', ',' , 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,2};
            [Total_Data, ~] = size(feat);
            label = zeros(Total_Data,1);
            for i = 1:Total_Data
                if isequal(C{1,1}{i},'B')
                    label(i)= 1;
                elseif isequal(C{1,1}{i},'L')
                    label(i)= 2;
                elseif isequal(C{1,1}{i},'R')
                    label(i)= 3;
                end
            end
            dataset_name = '29 Balance Scale' ;
         case 30
            % Climate Model Simulation Crashes Data Set
            % https://archive.ics.uci.edu/ml/datasets/Climate+Model+Simulation+Crashes
            disp('Loading DS 30 - Climate Model Simulation Crashes Data Set')
            fileID = fopen(fullfile(dataset_dir, 'pop_failures.dat'));
            format = [repmat('%f', [1, 21])];
            C = textscan(fileID, format, 'EmptyValue', -Inf, 'headerlines', 1, 'CollectOutput', 1);
            fclose(fileID);
            feat = C{1,1}(1:end, 3:end-1);
            label = C{1,1}(1:end, end);
            dataset_name = '30 Climate Model Simulation Crashes' ;
         case 31
            % DARWIN Data Set
            % https://archive.ics.uci.edu/dataset/732/darwin
            disp('Loading DS 31 - Darwin Data Set')
            C = readtable(fullfile(dataset_dir, 'DARWIN.csv')); % Stores data in a table
            feat = table2array(C(1:end, 2:end-1));
            [Total_Data, ~] = size(feat);
            label_Temp = C(1:end, end);
            label = zeros(Total_Data,1);
            for i = 1:Total_Data
                if isequal(label_Temp{i, 1}, {'P'})
                    label(i) = 1;
                elseif isequal(label_Temp{i, 1}, {'H'})
                    label(i) = 2;
                end
            end
            dataset_name = '31 DARWIN' ;
        case 32
            % Person Classification Gait Data Set
            % https://archive.ics.uci.edu/dataset/561/person+classification+gait+data
            % Generalization setting (Reviewer 4, item 2): in this dataset "person = class" (16 persons = 16 classes, 3 captures per person).
            % Splitting by group (person) would hold out entire classes -> "unseen-class recognition", which changes the task.
            % Therefore group is DELIBERATELY not set here; sample-level stratified split is used (closed-set recognition of the same 16 persons).
            disp('Loading DS 32 - Person Classification Gait Data Set')
            filename = fullfile(dataset_dir, "PersonGaitDataSet.mat");
            myVars = {"X","Y"};
            S = load(filename,myVars{:});
            feat = S.X;
            label = S.Y;
            dataset_name = '32 Person Classification Gait' ;
         case 33
            % Gastrointestinal Lesions in Regular Colonoscopy Data Set
            % https://archive.ics.uci.edu/dataset/408/gastrointestinal+lesions+in+regular+colonoscopy
            disp('Loading DS 33 - Gastrointestinal Lesions in Regular Colonoscopy Data Set')
            fileID = fopen(fullfile(dataset_dir, 'gastrointestinal+lesions+in+regular+colonoscopy.txt'));
            format = [repmat('%f', [1, 152])];
            C = textscan(fileID, format, 'Delimiter', ',' , 'headerlines', 1, 'CollectOutput', 1);
            fclose(fileID);
            feat = (C{1,1}(3:end, 1:end)).';
            label = (C{1,1}(1, 1:end)).';
            dataset_name = '33 Gastrointestinal Lesions in Regular Colonoscopy' ;
         case 34
            % Parkinson's Disease Classification Data Set
            % https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification
            disp('Loading DS 34 - Parkinsons Disease classification Data Set')
            C = readtable(fullfile(dataset_dir, 'pd_speech_features.csv')); % Stores data in a table
            feat = table2array(C(1:end, 3:end-1));
            label = table2array(C(1:end, end));
            group = table2array(C(1:end, 1));   % Column 1 is patient ID; 3 recordings per patient, used for group-level splitting
            dataset_name = '34 Parkinsons Disease Classification' ;
        case 35
            % TUNADROMD Data Set
            % https://archive.ics.uci.edu/dataset/813/tunadromd
            disp('Loading DS 35 - TUNADROMD Data Set')
            C = readtable(fullfile(dataset_dir, 'TUANDROMD.csv')); % Stores data in a table
            feat = table2array(C(1:end, 1:end-1));
            label = table2array(C(1:end, end));
            dataset_name = '35 TUNADROMD' ;
         case 36
            % Ultrasonic flowmeter diagnostics (Meter C and D)
            % https://archive.ics.uci.edu/dataset/433/ultrasonic+flowmeter+diagnostics
            disp('Loading DS 36 - Ultrasonic flowmeter diagnostics Data Set (Meter C and D)')
            % load('Meter A')
            % load('Meter B')
            load(fullfile(dataset_dir, 'Meter C'))
            load(fullfile(dataset_dir, 'Meter D'))
            C = [Meter_C; Meter_D];
            feat = (C(1:end, 1:end-1));
            label = (C(1:end, end));
            dataset_name = '36 Ultrasonic flowmeter diagnostics' ;
    
        case 37
            % https://jundongl.github.io/scikit-feature/datasets.html
            disp('Loading DS 37 GLI_85')
            load(fullfile(dataset_dir, 'GLI_85.mat')); 
            feat = X;
            label = Y; 
            
            if min(label) == 1 && max(label) == 2
                label(label == 1) = 0;
                label(label == 2) = 1;
            end
            
            dataset_name = '37 GLI_85';
        case 38
            % https://jundongl.github.io/scikit-feature/datasets.html
            disp('Loading DS 38 COIL20')
            load(fullfile(dataset_dir, 'COIL20.mat')); 
            feat = X;
            label = Y; 
            
            if min(label) == 1 && max(label) == 2
                label(label == 1) = 0;
                label(label == 2) = 1;
            end
            
            dataset_name = '38 COIL20';
        case 39
            % http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html
            % Generalization setting (Reviewer 4, item 2): in ORL "person = class" (40 persons = 40 classes).
            % No group is set; sample-level stratified split is used (closed-set face recognition); splitting by person would hold out unseen classes.
            disp('Loading DS 39 ORL')
            load(fullfile(dataset_dir, 'ORL.mat')); 
            feat = X;
            label = Y; 
            dataset_name = '39 ORL';
        case 40
            % https://jundongl.github.io/scikit-feature/datasets.html
            disp('Loading DS 40 warpPIE10P')
            load(fullfile(dataset_dir, 'warpPIE10P.mat')); 
            feat = X;
            label = Y; 
            dataset_name = '40 warpPIE10P';  
        case 41
            % https://jundongl.github.io/scikit-feature/datasets.html
            disp('Loading DS 41 warpAR10P')
            load(fullfile(dataset_dir, 'warpAR10P.mat')); 
            feat = X;
            label = Y; 
            dataset_name = '41 warpAR10P';   
        case 42 
            disp('Loading DS 42 - Hepatitis Data Set')
            
            % 1. Smart reading of data containing '?'
            % Use detectImportOptions to auto-detect the format and mark '?' as NaN
            filename = fullfile(dataset_dir, 'hepatitis.data');
            if exist(filename, 'file')
                opts = detectImportOptions(filename, 'FileType', 'text');
                % Tell MATLAB that '?' represents missing values
                opts = setvaropts(opts, opts.VariableNames, 'TreatAsMissing', '?');
                T = readtable(filename, opts);
                raw_data = table2array(T);
            else
                error('找不到文件 hepatitis.data，请确保文件在当前目录下');
            end
            
            % 2. Missing value handling
            % Missing values ('?') are already converted to NaN by readtable; do NOT impute globally here;
            % Imputation is done by common.preprocessData after the outer split, fitted on the training set only (prevent leakage).
            
            % 3. Separate features and labels
            % UCI Hepatitis: column 1 is Class, columns 2-20 are features
            Y_raw = raw_data(:, 1);
            feat = raw_data(:, 2:end);
            
            % 4. Label encoding
            % Original labels: 1 = DIE (died), 2 = LIVE (survived)
            % Conversion logic: focus on the minority class (died), set to 1; survived set to 0
            [Total_Data, ~] = size(feat);
            label = zeros(Total_Data, 1);
            
            % Keep 1 (DIE) as 1, change 2 (LIVE) to 0
            label(Y_raw == 1) = 1; 
            label(Y_raw == 2) = 0; 
            
            dataset_name = '42 Hepatitis';
        case 43
            % https://archive.ics.uci.edu/dataset/74/musk+clean2
            disp('Loading DS 43 Musk (Version 1)')
            fileID = fopen(fullfile(dataset_dir, 'clean1.data'));            
            format = ['%s %s' repmat('%f', [1, 167])];
            C = textscan(fileID, format, 'Delimiter', ',', 'CollectOutput', 1);
            fclose(fileID);
            
            feat = C{1,2}(:, 1:166);
            label = C{1,2}(:, 167);
            group = C{1}(:, 1);   % column 1 is the molecule name (e.g., MUSK-188); multiple conformers per molecule, used for group-level splitting
            dataset_name = '43 Musk (Version 1)';

        case 44 
            % https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications
            disp('Loading DS 44 - Myocardial infarction complications')
            
            filename = fullfile(dataset_dir, 'MI.data');
            
            % Read data (same as above)
            T = readtable(filename, 'FileType', 'text', ...
                          'ReadVariableNames', false, ...
                          'TreatAsMissing', '?');
            data = table2array(T);
            feat = data(:, 2:112);  % take exactly 111 features
            label = data(:, 124);

            dataset_name = '44 Myocardial infarction complications';
        
        case 45
            % https://archive.ics.uci.edu/dataset/1104/drug_induced_autoimmunity_prediction
            disp('Loading DS 45 - Drug Induced Autoimmunity Prediction')

            trainFile = fullfile(dataset_dir, 'DIA_trainingset_RDKit_descriptors.csv');
            testFile  = fullfile(dataset_dir, 'DIA_testset_RDKit_descriptors.csv');

            T_train = readtable(trainFile);
            T_test  = readtable(testFile);
            T_all   = [T_train; T_test];   % combine train + test

            feat = table2array(T_all(:, 3:end));
            col_names = T_all.Properties.VariableNames(3:end);
            dup_idx = strcmp(col_names, 'MaxEStateIndex');
            feat(:, dup_idx) = [];

            % Extract labels
            label = T_all.Label;

            dataset_name = '45 Drug Induced Autoimmunity Prediction';

        case 46
            % Urban Land Cover Data Set
            % https://archive.ics.uci.edu/ml/datasets/Urban+Land+Cover
            % Combine training (168) + testing (507) = 675 samples, 147 features
            disp('Loading DS 46 - Urban Land Cover')

            trainFile = fullfile(dataset_dir, 'urban_training.csv');
            testFile  = fullfile(dataset_dir, 'urban_testing.csv');

            T_train = readtable(trainFile);
            T_test  = readtable(testFile);
            T_all   = [T_train; T_test];   % combine train + test

            % Extract features (feat)
            % Column 1 is the text label ('class'), columns 2 to end are numeric features
            feat = table2array(T_all(:, 2:end));

            % Extract labels (label)
            raw_labels = T_all{:, 1};

            % Use grp2idx to convert text labels (e.g., 'car', 'tree') to numeric values (1, 2, 3...)
            [label, label_names] = grp2idx(raw_labels);

            dataset_name = '46 Urban Land Cover';
        case 47 
            disp('Loading DS 47 - Hill Valley (No Noise)')
            filename = fullfile(dataset_dir, 'Hill_Valley_without_noise_Training.data');
            
            T = readtable(filename, 'FileType', 'text');
            feat = table2array(T(:, 1:end-1));
            label = T.class;
            
            dataset_name = '47 Hill Valley';
        case 48 
            % https://github.com/jundongl/scikit-feature/tree/master/skfeature/data
            disp('Loading DS 48 colon')
            load(fullfile(dataset_dir, 'colon.mat')); 
            feat = X;
            label = Y; 
            dataset_name = '48 colon';  
        case 49 
            % Dataset: Internet Advertisements
            % URL: https://archive.ics.uci.edu/dataset/51/internet+advertisements
            disp('Loading DS 49 Internet Advertisements');
            
            opts = detectImportOptions(fullfile(dataset_dir, 'ad.data'), 'FileType', 'text');
            opts.MissingRule = 'fill'; 
            T = readtable(fullfile(dataset_dir, 'ad.data'), opts);
            
            raw_data = table2cell(T);
            feat_raw = raw_data(:, 1:end-1);
            label_raw = raw_data(:, end);

            feat = zeros(size(feat_raw));
            for j = 1:size(feat_raw, 2)
                col = feat_raw(:, j);
                if iscellstr(col) || isstring(col)
                    num_col = str2double(col);
                else
                    num_col = cell2mat(col);
                end
                % Keep missing values as NaN; do not impute globally with the full-data mean;
                % Imputation is done by common.preprocessData after the outer split, fitted on the training set only (prevent leakage).
                feat(:, j) = num_col;
            end
            
            % 4. Label handling (convert to 0 and 1)
            % Original labels are the strings 'ad.' and 'nonad.'
            label = zeros(length(label_raw), 1);
            for i = 1:length(label_raw)
                if contains(label_raw{i}, 'ad') && ~contains(label_raw{i}, 'nonad')
                    label(i) = 1; % advertisement
                else
                    label(i) = 0; % non-advertisement
                end
            end
            
            % Follow the existing code convention; ensure label range is as expected
            if min(label) == 1 && max(label) == 2
                label(label == 1) = 0;
                label(label == 2) = 1;
            end
            
            dataset_name = '49 Internet Ads';
    end   
end























