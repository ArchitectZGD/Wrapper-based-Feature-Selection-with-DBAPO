% Notation: This fitness function is for demonstration 

function [cost, error] = fitnessFunction(feat, label, X, HO, thres, category)
    % When global classifier is set in config.m, override the hard-coded 'KNN' in all algorithms
    global classifier
    if ~isempty(classifier)
        category = classifier;
    end

    jwrapperKNN = @fitness.jwrapperKNN ;
    jwrapperSVM = @fitness.jwrapperSVM ;
    jwrapperLR  = @fitness.jwrapperLR ;
    jwrapperMLP = @fitness.jwrapperMLP ;
    jwrapperRF  = @fitness.jwrapperRF ;
    random_select_mapping_linear = @tool_alogrithm.common.random_select_mapping_linear ;

    [~, Total_Feature] = size(X);
    X = random_select_mapping_linear(X, Total_Feature, thres) ;
    
    if sum(X == 1) == 0 
      cost = 1; 
      error = 1;
    else 
      s_Feat_index = find(X == 1);
      s_Feat = feat(:, s_Feat_index);
      switch upper(category)
          case 'KNN'
              error = jwrapperKNN(s_Feat, label, HO);
          case 'SVM'
              error = jwrapperSVM(s_Feat, label, HO);
          case 'LR'
              error = jwrapperLR(s_Feat, label, HO);
          case 'MLP'
              error = jwrapperMLP(s_Feat, label, HO);
          case 'RF'
              error = jwrapperRF(s_Feat, label, HO);
      end 

      s_Feat_ratio = length(s_Feat_index)/length(X);
      global alpha
      if isempty(alpha)
          alpha = 0.99;   % default weight; DBAPO_C8 sets it to 0.95
      end
      cost = alpha*error + (1-alpha)*s_Feat_ratio;
    end
end
