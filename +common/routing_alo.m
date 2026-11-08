function [sfeature, Sf, Nf, curve, ErrorCurve, diagnostics, NfCurve] = routing_alo(Algorithm_Name,feature,label,N,max_FEs,HO,CR, classifier_name, fitness_alpha)
    % Declaring globals is forbidden inside a parfor/spmd body, but allowed in a normal function called by parfor.
    % Set the global classifier once on the worker at the start of each task.
    if nargin >= 8 && ~isempty(classifier_name)
        global classifier
        classifier = classifier_name;
    end

    % Set global alpha (fitness weight) at the start of each task; default 0.99, can be overridden via fitness_alpha
    global alpha
    if nargin >= 9 && ~isempty(fitness_alpha)
        alpha = fitness_alpha;
    else
        alpha = 0.99;
    end

    diagnostics = [];   % only DBAPO populates this (ades_log / hamming_log / db_diag_log, etc.)
    NfCurve = [];       % only DBAPO populates this (true feature count per generation, Reviewer 4 item 6)

    [current_folder, ~, ~] = fileparts(mfilename('fullpath'));
    algorithm_path = fullfile(current_folder, '..', 'Algorithm', Algorithm_Name);
    addpath(algorithm_path);
    
    switch Algorithm_Name

        case 'APO' % (56) Arctic Puffin Optimization (APO)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jAPO(feature,label,N,max_FEs,HO);
        case 'DBAPO' % 66 (enhanced DBAPO with diagnostics)
            [sfeature,Sf,Nf,curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO(feature,label,N,max_FEs,HO);
        case 'DBAPO_C2' % OFAT: rho_init=0.6
            [sfeature,Sf,Nf,curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO_C2(feature,label,N,max_FEs,HO);
        case 'DBAPO_C3' % OFAT: rho_min=0.4
            [sfeature,Sf,Nf,curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO_C3(feature,label,N,max_FEs,HO);
        case 'DBAPO_C4' % OFAT: T_low=30
            [sfeature,Sf,Nf,curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO_C4(feature,label,N,max_FEs,HO);
        case 'DBAPO_C5' % OFAT: T_mid=1000
            [sfeature,Sf,Nf,curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO_C5(feature,label,N,max_FEs,HO);
        case 'DBAPO_C6' % OFAT: tau_stag=3
            [sfeature,Sf,Nf,curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO_C6(feature,label,N,max_FEs,HO);
        case 'DBAPO_C7' % OFAT: tau_stag=10
            [sfeature,Sf,Nf,curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO_C7(feature,label,N,max_FEs,HO);
        case 'DBAPO_A2' % ablation: only ADES (no DB)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jDBAPO_A2(feature,label,N,max_FEs,HO);
        case 'DBAPO_A3' % ablation: only DB (no ADES)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jDBAPO_A3(feature,label,N,max_FEs,HO);
        case 'DBAPO_A4' % ablation: no elite-preserving survivor update
            [sfeature,Sf,Nf,curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO_A4(feature,label,N,max_FEs,HO);
        case 'DBAPO_A5' % ablation: fixed trigger (no ATM)
            [sfeature,Sf,Nf,curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO_A5(feature,label,N,max_FEs,HO);
        case 'DBAPO_B1' % direct binary recombination
            [sfeature,Sf,Nf,curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO_B1(feature,label,N,max_FEs,HO);
        case 'DBAPO_C8' % OFAT: alpha=0.95
            [sfeature,Sf,Nf,curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO_C8(feature,label,N,max_FEs,HO);
        case 'DBAPO_C9' % OFAT: dim boundaries 100/500
            [sfeature,Sf,Nf,curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO_C9(feature,label,N,max_FEs,HO);
        case 'APO_Q' % 67 
            [sfeature,Sf,Nf,curve, ErrorCurve] = jAPO_Q(feature,label,N,max_FEs,HO);
        case 'CEAM_APO' % 67 
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCEAM_APO(feature,label,N,max_FEs,HO);

        case 'MRMR' % 71
            [sfeature,Sf,Nf,curve, ErrorCurve] = jMRMR(feature,label,N,max_FEs,HO);    
        case 'HO' % 70
            [sfeature,Sf,Nf,curve, ErrorCurve] = jHO(feature,label,N,max_FEs,HO); 
        case 'WSO' % 69 
            [sfeature,Sf,Nf,curve, ErrorCurve] = jWSO(feature,label,N,max_FEs,HO);         
        case 'SFOA' % 67 
            [sfeature,Sf,Nf,curve, ErrorCurve] = jSFOA(feature,label,N,max_FEs,HO);
        case 'GWO' % 68 
            [sfeature,Sf,Nf,curve, ErrorCurve] = jGWO(feature,label,N,max_FEs,HO);
        case 'APO_GA_Parallel' % 64 
            [sfeature,Sf,Nf,curve, ErrorCurve] = jAPO_GA_Parallel(feature,label,N,max_FEs,HO);
        case 'CO-DE' % (1) Algorithm_Name = Chaotic Oppositional Differential Evolution
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCO_DE(feature,label,N,max_FEs,CR,HO);
        case 'COHDEPSO' % (2) Algorithm_Name = Chaotic Oppositional based Hybridized DE with PSO
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCO_HDEPSO(feature,label,N,max_FEs,CR,HO);
        case 'PSO' % (3) Algorithm_Name = Particle Swarm Optimization
            [sfeature,Sf,Nf,curve, ErrorCurve] = jPSO(feature,label,N,max_FEs,HO);
        case 'OSEPSO' % (4) Algorithm_Name = Oppositional and Social Learning with Enhance Operator incorporate into PSO
            [sfeature,Sf,Nf,curve, ErrorCurve] = jOSEPSO(feature,label,N,max_FEs,HO);
        case 'AOA' % (5) Algorithm_Name = Arithmetic Optimization Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jAOA(feature,label,N,max_FEs,HO);
        case 'BLPSO' % (6) Algorithm_Name = Biogeography-Based Learning Particle Swarm Optimization
            [sfeature,Sf,Nf,curve, ErrorCurve] = jBLPSO(feature,label,N,max_FEs,HO);
        case 'CDEEPSO' % (7) Algorithm_Name = Hybrid Canonical Differential Evolutionary PSO
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCDEEPSO(feature,label,N,max_FEs,HO);
        case 'CHIO' % (8) Algorithm_Name = Coronavirus Herd Immunity Optimizer
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCHIO(feature,label,N,max_FEs,HO);
        case 'EO' % (9) Algorithm_Name = Equilibrium Optimizer
            [sfeature,Sf,Nf,curve, ErrorCurve] = jEO(feature,label,N,max_FEs,HO);
        case 'FDA' % (10) Algorithm_Name = Flow Direction Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jFDA(feature,label,N,max_FEs,HO);
        case 'GNDO' % (11) Algorithm_Name = Generalized Normal Distribution Optimization
            [sfeature,Sf,Nf,curve, ErrorCurve] = jGNDO(feature,label,N,max_FEs,HO);
        case 'HGSO' % (12) Algorithm_Name = Henry Gas Solubility Optimization
            [sfeature,Sf,Nf,curve, ErrorCurve] = jHGSO(feature,label,N,max_FEs,HO);
        case 'MPA' % (13) Algorithm_Name = Marine Predators Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jMPA(feature,label,N,max_FEs,HO);
        case 'MVO' % (14) Algorithm_Name = Multi-Verse Optimizer
            [sfeature,Sf,Nf,curve, ErrorCurve] = jMVO(feature,label,N,max_FEs,HO);
        case 'SCA' % (15) Algorithm_Name = Sine Cosine Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jSCA(feature,label,N,max_FEs,HO);
        case 'SSA' % (16) Algorithm_Name = Salp Swarm Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jSSA(feature,label,N,max_FEs,HO);
        case 'AHA' % (17) Algorithm_Name = Artificial Hummingbird Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jAHA(feature,label,N,max_FEs,HO);
        case 'AVOA' % (18) Algorithm_Name = African Vulture Optimization Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jAVOA(feature,label,N,max_FEs,HO);
        case 'GTO' % (19) Algorithm_Name = Gorilla Troops Optimizer Algorithm (2021)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jGTO(feature,label,N,max_FEs,HO);
        case 'MTDE' % (20) Algorithm_Name = Multi-Trial vector-based Differential Evolution
            [sfeature,Sf,Nf,curve, ErrorCurve] = jMTDE(feature,label,N,max_FEs,HO);
        case 'BeSD' % (21) Algorithm_Name = Bezier Search Differential Evolution Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jBeSD(feature,label,N,max_FEs,HO);
        case 'WDE' % (22) Algorithm_Name = Weighted Differential Evolution Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jWDE(feature,label,N,max_FEs,HO);
        case 'DSSA' % (23) Algorithm_Name = Differential Squirrel Search Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jDSSA(feature,label,N,max_FEs,HO);
        case 'FDBAGDE' % (24) Algorithm_Name = Differential Squirrel Search Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jFDBAGDE(feature,label,N,max_FEs,HO);
        case 'FSTDE' % (25) Algorithm_Name = Fuzzy Self-Tuning Differential Evolution Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jFSTDE(feature,label,N,max_FEs,HO);
        case 'HyDEDF' % (26) Algorithm_Name = Hybrid-Adaptive Differential Evolution with Decay Function (HyDE-DF) Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jHyDEDF(feature,label,N,max_FEs,HO);
        case 'SSO' % (27) Sperm Swarm Optimization Algorithm (2018)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jSSO(feature,label,N,max_FEs,HO);
        case 'CO' % (28) Cheetah Optimizer (CO) (2022)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCO(feature,label,N,max_FEs,HO);      
        case 'ODSFMFO' % (29) Opposition Death Mechanism Shuffled Frog and Moth ...
            % Flame Optimization with Improved Flame Generation Mechanism (2021)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jODSFMFO(feature,label,N,max_FEs,HO);
        case 'CFDA' % (30) Chaotic Flow Direction Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCFDA(feature,label,N,max_FEs,HO);
        case 'CAVOA' % (31) Chaotic African Vulture Optimization Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCAVOA(feature,label,N,max_FEs,HO);
        case 'CAVOA1' % (32) Chaotic Circle African Vulture Optimization Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCAVOA1(feature,label,N,max_FEs,HO);
        case 'CFDA1' % (33) Chaotic Circle Flow Direction Algorithm
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCFDA1(feature,label,N,max_FEs,HO);
        case 'CETLBO V2' % (34) 
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCETLBO_V2(feature,label,N,max_FEs,HO);
        case 'CTLBO' % (35) Converged Teaching–Learning–based Optimization
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCTLBO(feature,label,N,max_FEs,HO);
        case 'TLBOPSO' % (36) Teaching-Learning-Based Optimization with the social character of PSO
            [sfeature,Sf,Nf,curve, ErrorCurve] = jTLBOPSO(feature,label,N,max_FEs,HO);
        case 'TLBO' % (37) Teaching-Learning-Based Optimization
            [sfeature,Sf,Nf,curve, ErrorCurve] = jTLBO(feature,label,N,max_FEs,HO);
        case 'TTAO' % (38) Triangulation Topology Aggregation Optimizer (2023)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jTTAO(feature,label,N,max_FEs,HO);
        case 'DO' % (39) Dandelion Optimizer (2022)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jDO(feature,label,N,max_FEs,HO);
        case 'DMOA' % (40) Dwarf Mongoose Optimization Algorithm (2022)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jDMOA(feature,label,N,max_FEs,HO);
        case 'CPO' % (41) Chinese Pangolin Optimizer (2024) 
            [sfeature,Sf,Nf,curve, ErrorCurve] = jCPO(feature,label,N,max_FEs,HO);
        case 'AO' % (42) Aquila Optimizer (2021)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jAO(feature,label,N,max_FEs,HO);
        case 'ZOA' % (43) Zebra optimization algorithm (2023)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jZOA(feature,label,N,max_FEs,HO);
        case 'WOA' % (44) Walrus Optimization Algorithm (2023)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jWOA(feature,label,N,max_FEs,HO);
        case 'GOA' % (45) Golf Optimization Algorithm (2023)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jGOA(feature,label,N,max_FEs,HO);
        case 'RTH' % (46) Red-Tailed Hawk Algorithm (2023)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jRTH(feature,label,N,max_FEs,HO);
        case 'SHO' % (47) Sea Horse Optimizer Algorithm (2023)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jSHO(feature,label,N,max_FEs,HO);
        case 'OOA' % (48) Osprey Optimization Algorithm (2023)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jOOA(feature,label,N,max_FEs,HO);
        case 'DTBO' % (49) Driving Training-Based Optimization Algorithm (2022)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jDTBO(feature,label,N,max_FEs,HO);
        case 'PO' % (50) Puma Optimizer Algorithm (2024)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jPO(feature,label,N,max_FEs,HO);
        case 'NIWTLBO' % (51) 
            [sfeature,Sf,Nf,curve, ErrorCurve] = jNIWTLBO(feature,label,N,max_FEs,HO);
        case 'LMRTLBO' % (52) 
            [sfeature,Sf,Nf,curve, ErrorCurve] = jLMRTLBO(feature,label,N,max_FEs,HO);
        case 'DhoH' % (53) Dholes Hunting (2024)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jDhoH(feature,label,N,max_FEs,HO);
        case 'EVO' % (54) Energy Valley Optimizer (2023)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jEVO(feature,label,N,max_FEs,HO);
        case 'ILCO' % (55) Improved Life Cycle Optimization(2024)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jILCO(feature,label,N,max_FEs,HO);
        case 'APO_Parallel_SVM' % (58) Parallel version of APO + SVM
            [sfeature,Sf,Nf,curve, ErrorCurve] = jAPO_parallel(feature,label,N,max_FEs,HO);
        case 'APO_Parallel_RF' % (59) Parallel version of APO + RF
            [sfeature,Sf,Nf,curve, ErrorCurve] = jAPO_parallel(feature,label,N,max_FEs,HO);
        case 'APO_Parallel_LR' % (60) Parallel version of APO + LR
            [sfeature,Sf,Nf,curve, ErrorCurve] = jAPO_parallel(feature,label,N,max_FEs,HO);
        case 'APO_Parallel_MLP' % (61) Parallel version of APO + MLP
            [sfeature,Sf,Nf,curve, ErrorCurve] = jAPO_parallel(feature,label,N,max_FEs,HO);    
        case 'APO_Parallel_v2' % (62) Parallel version of APO (full)
            [sfeature,Sf,Nf,curve, ErrorCurve] = jAPO_parallel_full(feature,label,N,max_FEs,HO);
        
    end
end 