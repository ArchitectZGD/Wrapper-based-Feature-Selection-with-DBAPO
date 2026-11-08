function [Mean_Error, StdDev_Error, Mean_Acc, StdDev_Acc, Ave_Nf, Ratio_NF, Ave_Time, All_Fitness, Ave_Fitness] ...
    = basedAnalysis(Algorithm_Name, DataSet_Index, curve_All, Nf_All, feature, ErrorCurve_All, Time_Per_Run)
    % (1) Mean classification error
    Mean_Error = mean(ErrorCurve_All(:,end));
    % (2) Standard deviation of classification error
    StdDev_Error = std(ErrorCurve_All(:,end));
    % (3) Mean classification accuracy
    Mean_Acc = mean(1-ErrorCurve_All(:,end));
    % (4) Standard deviation of classification accurary
    StdDev_Acc = std(1-ErrorCurve_All(:,end));
    % (5) Average number of selected features
    Ave_Nf = mean(Nf_All);
    % (6) Ratio of feature being selected
    Ratio_NF = Ave_Nf/size(feature,2);
    % (7) Average time to complete 
    Ave_Time = mean(Time_Per_Run);
    % (7) Average fitness
    All_Fitness = curve_All(:,end);
    Ave_Fitness = mean(All_Fitness);
    %clc
    disp(['Summary of results by ' Algorithm_Name ' for Dataset = ' num2str(DataSet_Index)])
    disp(['Mean error = '  num2str(Mean_Error)])
    disp(['StdDev_Error = '  num2str(StdDev_Error)])
    disp(['Mean_Acc = '  num2str(Mean_Acc)])
    disp(['StdDev_Acc = '  num2str(StdDev_Acc)])
    disp(['Average Selected features = '  num2str(Ave_Nf)])
    disp(['Total features = '  num2str(size(feature,2))])
    disp(['Ratio_Nf = '  num2str(Ratio_NF)])
    disp(['Average Time = '  num2str(Ave_Time)])
    disp(['Average Fitness = '  num2str(Ave_Fitness)])
end