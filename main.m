clc
clear
close all  

DataSet_Index = [9];
Algorithms = ["DBAPO"] ;

Main(Algorithms, DataSet_Index) ; 
clearvars -except i


function [] = Main(Algorithms, DataSet_Index)
    routing_alo = @common.routing_alo ;
    Load_UCI_Data = @common.Load_UCI_Data ;
    basedAnalysis = @common.basedAnalysis ; 
    graphPlotting = @common.graphPlotting ; 
    generateOverView = @common.generateOverView ; 
    generateStatistics = @common.generateStatistics ; 

    run(fullfile(fileparts(mfilename('fullpath')), 'config.m'));
    addpath(fullfile(fileparts(mfilename('fullpath')), 'Dataset'));

    for Algorithm_Name = Algorithms
        disp(['------Algorithm_Name: ', Algorithm_Name,'------'])
        for dataindex = DataSet_Index
            [feature, label, dataset_name, group] = Load_UCI_Data(dataindex);
            % Untouched master copy: the held-out TEST set is carved from this
            % every run and is NEVER seen by the optimiser (data-leakage fix).
            feat_full  = feature;
            label_full = label;
            output_key = [Algorithm_Name, '-', dataset_name, '-'] ; 

            % Per-run stores (three-layer validation).
            ConfMat_All     = cell(max_Run,1);
            Metrics_All     = cell(max_Run,1);
            TestAcc_All     = zeros(max_Run,1);
            BalancedAcc_All = zeros(max_Run,1);
            MacroF1_All     = zeros(max_Run,1);
            MacroPrec_All   = zeros(max_Run,1);
            MacroRecall_All = zeros(max_Run,1);
            TestIdx_All     = cell(max_Run,1);
            InnerScheme_All = cell(max_Run,1);
            Diagnostics_All = cell(max_Run,1);

            for r = 1: max_Run
                tic
                % ---------------------------------------------------------
                % Three-layer (nested) validation:
                %   1) OUTER split -> TRAINING + held-out TEST (group-aware)
                %   2) INNER partition -> drawn from TRAINING only (fitness)
                %   3) FINAL eval -> on the held-out TEST set, once
                % Same seed for every algorithm -> paired tests stay valid.
                % ---------------------------------------------------------
                seed = 1000*dataindex + r;
                rng(seed);   % cvpartition/randperm ignore the seed argument, so seed the global stream explicitly
                [trIdx, teIdx] = common.makeOuterSplit(label_full, group, ho, seed);
                featTr  = feat_full(trIdx, :);   labelTr = label_full(trIdx, :);
                featTe  = feat_full(teIdx, :);   labelTe = label_full(teIdx, :);

                % Preprocessing (missing-value imputation): fit on training set only, then apply to test set (leakage prevention)
                [featTr, featTe] = common.preprocessData(featTr, featTe);

                % Inner partition: group-wise split when group is available (same molecule/patient stays on one side)
                groupTr = [];
                if ~isempty(group), groupTr = group(trIdx); end
                rng(seed + 1);   % inner partition consumes its own deterministic stream
                [HO, InnerScheme_All{r}] = common.buildInnerPartition(labelTr, ho_inner, groupTr, seed + 1);

                % Search uses TRAINING data + INNER partition only.
                [sfeature, Sf, Nf, curve, ErrorCurve, diagnostics, NfCurve] = routing_alo(Algorithm_Name, featTr, labelTr, N, max_FEs, HO, CR);
                Diagnostics_All{r} = diagnostics;

                % Final evaluation on the held-out TEST set (headline number).
                [ConfMat_All{r}, TestAcc_All(r), ClassOrder, Metrics_All{r}] = ...
                    common.jFinalEval(featTr, labelTr, featTe, labelTe, Sf);
                BalancedAcc_All(r) = Metrics_All{r}.balancedAcc;
                MacroF1_All(r)     = Metrics_All{r}.macroF1;
                MacroPrec_All(r)   = Metrics_All{r}.macroPrec;
                MacroRecall_All(r) = Metrics_All{r}.macroRecall;

                Time_Per_Run(r) = toc;
                Nf_All(r,1) = Nf;
                curve_All(r,:) = curve(end-max_FEs+1:end);
                ErrorCurve_All(r,:) = ErrorCurve(end-max_FEs+1:end);
                Sf_All{r} = Sf;
                TestIdx_All{r} = find(teIdx)';

                % feature subset size per FE (inner-validation trajectory)
                % Back-solve from the fitness objective: fitness = alpha*err + (1-alpha)*ratio,
                % so ratio = (fitness-alpha*err)/(1-alpha), equal to the true selected count of that generation's best solution.
                % round + clamping avoids bias from floating-point error and the no-feature penalty (cost=1).
                [~, total_feature]=size(featTr);
                if ~isempty(NfCurve)
                    featureSubsetSize_All(r,:) = NfCurve(end-max_FEs+1:end);  % actual recorded values
                else
                    alpha = 0.99;
                    ratio = (curve_All(r,:) - alpha*ErrorCurve_All(r,:)) / (1-alpha);
                    ratio = max(0, min(1, ratio));
                    featureSubsetSize_All(r,:) = round(total_feature * ratio);  % back-calculated (numerically equivalent)
                end
            end
            
            %% analysis (inner-validation trajectory metrics)
            [Mean_Error, StdDev_Error, Mean_Acc, StdDev_Acc, Ave_Nf, Ratio_NF, Ave_Time, All_Fitness, Ave_Fitness] ...
            = basedAnalysis(Algorithm_Name, dataindex, curve_All, Nf_All, feat_full, ErrorCurve_All, Time_Per_Run);

            %% HEADLINE metrics: held-out TEST set (leakage-free)
            Mean_TestAcc   = mean(TestAcc_All);
            StdDev_TestAcc = std(TestAcc_All);
            Mean_TestError = 1 - Mean_TestAcc;
            Mean_BalancedAcc = mean(BalancedAcc_All);
            Mean_MacroF1     = mean(MacroF1_All);
            Mean_MacroPrec   = mean(MacroPrec_All);
            Mean_MacroRecall = mean(MacroRecall_All);
            [~, best_run]  = min(ErrorCurve_All(:,end));   % best by INNER error, no test peeking
            Best_TestAcc   = TestAcc_All(best_run);

            disp(['Mean TEST accuracy (reported) = ' num2str(Mean_TestAcc)])
            disp(['StdDev TEST accuracy = ' num2str(StdDev_TestAcc)])

            %% Graph Plotting (inner-validation curves)
            graphPlotting(Algorithm_Name, dataset_name, curve_All, ErrorCurve_All, featureSubsetSize_All, max_FEs) ;
            
            %% Save the results of multiple run into a single mat file
            filename = string(Algorithm_Name) + "_" + string(max_Run) + ".mat";
            folderPath = fullfile(fileparts(mfilename('fullpath')), 'Data Result', dataset_name);
            if ~isfolder(folderPath)
                mkdir(folderPath);
            end
            filelocation = fullfile(folderPath ,filename) ; 
            save(filelocation,'Nf_All', 'curve_All', 'Sf_All', 'ErrorCurve_All', ...
                'Mean_Error', 'StdDev_Error', 'Mean_Acc', 'StdDev_Acc', 'Ave_Nf',...
                'Ratio_NF', 'Ave_Time', 'All_Fitness', 'Ave_Fitness', 'N', ...
                'max_FEs', "CR", 'max_Run', 'featureSubsetSize_All','total_feature', ...
                'TestAcc_All', 'ConfMat_All', 'ClassOrder', 'TestIdx_All', ...
                'Metrics_All', 'BalancedAcc_All', 'MacroF1_All', 'MacroPrec_All', 'MacroRecall_All', ...
                'Mean_TestAcc', 'StdDev_TestAcc', 'Mean_TestError', 'Best_TestAcc', ...
                'Mean_BalancedAcc', 'Mean_MacroF1', 'Mean_MacroPrec', 'Mean_MacroRecall', ...
                'InnerScheme_All', 'Diagnostics_All', 'ho', 'ho_inner');
            generateStatistics(folderPath, max_Run);
        end 
    end 
    generateOverView(max_Run, Algorithms);
end
