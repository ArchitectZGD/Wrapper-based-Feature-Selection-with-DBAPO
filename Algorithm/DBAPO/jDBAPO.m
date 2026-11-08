function [sfeature, Sf, Nf, curve, ErrorCurve, diagnostics, NfCurve] = jDBAPO(feature, label, PopSize, FEmax, HO)
% Enhanced DBAPO with reviewer fixes:
%   [R4.10]  ADES progress switched from iter/max_iter to fes/FEmax
%   [R4.11a] Hamming distance logging at each AECO trigger
%   [R4.11b] Adaptive stagnation threshold: (1-alpha)/Dim
%   [R4.10]  ADES trajectory monitoring log
%
% New output: diagnostics (struct) containing all monitoring data

fitnessfun = @fitness.fitnessFunction;
levyFlightVec = @tool_alogrithm.common.levyFlightVec;

Dim = size(feature, 2);
thres = 0.5;
fes = 0;

curve_hist = inf(1, FEmax);
ErrorCurve = inf(1, FEmax);
Nf_hist = zeros(1, FEmax);   % truly record the number of features selected by best solution per generation (Reviewer 4 item 6)

% --- APO parameters ---
F = 0.5;
alpha = 0.99;  % [R4.11b] explicitly define alpha for stagnation threshold

% --- Dual-population parameters ---
ade_params.Pop_total = PopSize;
ade_params.Pop_init = round(PopSize * 0.8);
ade_params.Pop_min = round(PopSize * 0.2);

f_best = inf;
f_second_best = inf;
x_best = zeros(1, Dim);
x_second_best = zeros(1, Dim);
e_best = inf;

stagnation_count = 0;
prev_f_best = inf;
stagnation_thres = 5;
prob_new_best = 0.5;
dim_low_threshold = 50;
dim_mid_threshold = 300;

% [R4.11b] Adaptive stagnation threshold based on alpha and Dim
% Physical meaning: fitness change smaller than removing one feature
stagnation_threshold = (1 - alpha) / Dim;

persistent_newbest_pending = false;
db_triggered_bins = false(1, 19);

ExploreSize = ade_params.Pop_init;
DevelopSize = ade_params.Pop_total - ExploreSize;

% =====================================================================
% [R4.10] ADES trajectory monitoring log
% =====================================================================
max_possible_iters = ceil(FEmax / 2);  % safe upper bound for pre-allocation
ades_log.iter            = zeros(1, max_possible_iters);
ades_log.fes             = zeros(1, max_possible_iters);
ades_log.ExploreSize     = zeros(1, max_possible_iters);
ades_log.DevelopSize     = zeros(1, max_possible_iters);
ades_log.progress_frac   = zeros(1, max_possible_iters);

% =====================================================================
% [R4.11a] Hamming distance logging
% =====================================================================
hamming_log.distances    = [];   % Hamming distance at each AECO trigger
hamming_log.fes_at_trigger = [];  % fes value when AECO was triggered
hamming_log.trigger_count = 0;    % total number of AECO triggers

% =====================================================================
% [R4.11a] AERS duplicate/unique evaluation diagnostics (Reviewer 4 item 9)
% =====================================================================
db_diag_log.total_evaluated  = [];   % total number of candidate masks evaluated per AERS trigger
db_diag_log.unique_evaluated = [];   % number of truly distinct masks after deduplication
db_diag_log.duplicate_count  = [];   % number of duplicate evaluations = total - unique
db_diag_log.fes_at_trigger   = [];   % fes at trigger time
db_diag_log.trigger_count    = 0;

% ----------------- Initialisation -----------------
x_total = rand(PopSize, Dim);
x_explore = x_total(1:ExploreSize, :);
x_develop = x_total((ExploreSize+1):end, :);

[fit_explore, Err_explore] = evaluatePopulation(x_explore, ExploreSize);
[fit_develop, Err_develop] = evaluatePopulation(x_develop, DevelopSize);

% ----------------- Main Loop -----------------
iter = 0;
max_iter = ceil(FEmax / PopSize);  % kept only as safety upper bound for while loop
while fes < FEmax && iter < max_iter
    iter = iter + 1;
    new_best_found = false;

    % =================================================================
    % [R4.10] ADES: Dynamic population resizing using fes/FEmax
    % =================================================================
    Pop_init = ade_params.Pop_init;
    Pop_min = ade_params.Pop_min;

    % --- CHANGED: use fes/FEmax instead of iter/max_iter ---
    progress_fraction = fes / FEmax;
    new_ExploreSize = round((Pop_min - Pop_init) * progress_fraction + Pop_init);
    new_ExploreSize = max(4, new_ExploreSize);
    new_ExploreSize = min(PopSize - 4, new_ExploreSize);

    % --- Migration bridge ---
    if new_ExploreSize < ExploreSize
        num_to_move = ExploreSize - new_ExploreSize;
        [~, explore_idx] = sort(fit_explore);

        best_to_migrate_x = x_explore(explore_idx(1:num_to_move), :);
        best_to_migrate_fit = fit_explore(explore_idx(1:num_to_move));
        best_to_migrate_err = Err_explore(explore_idx(1:num_to_move));

        x_develop = [x_develop; best_to_migrate_x];
        fit_develop = [fit_develop; best_to_migrate_fit];
        Err_develop = [Err_develop; best_to_migrate_err];

        worst_indices = explore_idx(end-num_to_move+1:end);
        x_explore(worst_indices, :) = [];
        fit_explore(worst_indices) = [];
        Err_explore(worst_indices) = [];

        ExploreSize = size(x_explore, 1);
        DevelopSize = size(x_develop, 1);
    end

    % =================================================================
    % [R4.10] Record ADES trajectory for this iteration
    % =================================================================
    ades_log.iter(iter)          = iter;
    ades_log.fes(iter)           = fes;
    ades_log.ExploreSize(iter)   = ExploreSize;
    ades_log.DevelopSize(iter)   = DevelopSize;
    ades_log.progress_frac(iter) = progress_fraction;

    % --------------- Exploration Phase (Aerial Flight) ---------------
    if ExploreSize > 0
        Y = zeros(ExploreSize, Dim);
        Z = zeros(ExploreSize, Dim);
        for i = 1:ExploreSize
            idx = setdiff(1:ExploreSize, i);
            r = idx(randi(length(idx)));
            LF = levyFlightVec(Dim);
            R = round(0.5 * (0.05 + rand)) * randn;
            S = tan((rand - 0.5) * pi);
            Y(i,:) = x_explore(i,:) + (x_explore(i,:) - x_explore(r,:)) .* LF + R;
            Z(i,:) = Y(i,:) * S;
            Y(i,:) = min(max(Y(i,:), 0), 1);
            Z(i,:) = min(max(Z(i,:), 0), 1);
        end

        candidates = [Y; Z];
        cand_fit = zeros(2*ExploreSize, 1);
        cand_err = zeros(2*ExploreSize, 1);

        for i = 1:2*ExploreSize
            if fes >= FEmax, break; end
            [cand_fit(i), cand_err(i)] = fitnessfun(feature, label, candidates(i,:), HO, thres, 'KNN');
            fes = fes + 1;
            if cand_fit(i) < f_best
                f_second_best = f_best;
                x_second_best = x_best;
                f_best = cand_fit(i);
                e_best = cand_err(i);
                x_best = candidates(i,:);
                new_best_found = true;
            elseif cand_fit(i) < f_second_best && cand_fit(i) > f_best
                f_second_best = cand_fit(i);
                x_second_best = candidates(i,:);
            end
            curve_hist(fes) = f_best;
            ErrorCurve(fes) = e_best;
            Nf_hist(fes) = numel(find(x_best > thres));
        end

        % Elite-preserving update (keep best N)
        all_x = [x_explore; candidates];
        all_fit = [fit_explore; cand_fit];
        all_err = [Err_explore; cand_err];
        [all_fit, idx] = sort(all_fit);
        x_explore = all_x(idx(1:ExploreSize), :);
        fit_explore = all_fit(1:ExploreSize);
        Err_explore = all_err(idx(1:ExploreSize));
    end

    % --------------- Exploitation Phase (Underwater Foraging) ---------------
    if DevelopSize > 0
        W = zeros(DevelopSize, Dim);
        Yw = zeros(DevelopSize, Dim);
        Zw = zeros(DevelopSize, Dim);
        for i = 1:DevelopSize
            idx = setdiff(1:DevelopSize, i);
            perm = idx(randperm(length(idx), 3));
            r1 = perm(1); r2 = perm(2); r3 = perm(3);
            LF = levyFlightVec(Dim);
            if rand() >= 0.5
                W(i,:) = x_develop(r1,:) + F*LF.*(x_develop(r2,:) - x_develop(r3,:));
            else
                W(i,:) = x_develop(r1,:) + F*(x_develop(r2,:) - x_develop(r3,:));
            end
            W(i,:) = min(max(W(i,:), 0), 1);

            f = 0.1 * (rand - 1) * (max_iter - iter)/max_iter;
            Yw(i,:) = W(i,:) * (1 + f);
            Yw(i,:) = min(max(Yw(i,:), 0), 1);

            perm2 = idx(randperm(length(idx), 2));
            r1p = perm2(1); r2p = perm2(2);
            if rand >= 0.5
                LF2 = levyFlightVec(Dim);
                Zw(i,:) = x_develop(i,:) + F*LF2.*(x_develop(r1p,:) - x_develop(r2p,:));
            else
                beta = rand;
                Zw(i,:) = x_develop(i,:) + beta*(x_develop(r1p,:) - x_develop(r2p,:));
            end
            Zw(i,:) = min(max(Zw(i,:), 0), 1);
        end

        candidates = [W; Yw; Zw];
        cand_fit = zeros(3*DevelopSize, 1);
        cand_err = zeros(3*DevelopSize, 1);

        for i = 1:3*DevelopSize
            if fes >= FEmax, break; end
            [cand_fit(i), cand_err(i)] = fitnessfun(feature, label, candidates(i,:), HO, thres, 'KNN');
            fes = fes + 1;

            if cand_fit(i) < f_best
                f_second_best = f_best;
                x_second_best = x_best;
                f_best = cand_fit(i);
                e_best = cand_err(i);
                x_best = candidates(i,:);
                new_best_found = true;
            elseif cand_fit(i) < f_second_best && cand_fit(i) > f_best
                f_second_best = cand_fit(i);
                x_second_best = candidates(i,:);
            end
            curve_hist(fes) = f_best;
            ErrorCurve(fes) = e_best;
            Nf_hist(fes) = numel(find(x_best > thres));
        end

        % Elite-preserving update (keep best N)
        all_x = [x_develop; candidates];
        all_fit = [fit_develop; cand_fit];
        all_err = [Err_develop; cand_err];
        [all_fit, idx] = sort(all_fit);
        x_develop = all_x(idx(1:DevelopSize), :);
        fit_develop = all_fit(1:DevelopSize);
        Err_develop = all_err(idx(1:DevelopSize));
    end

    % --------------- AECO (ATM + AERS) ---------------
    has_reinforce_right = new_best_found || persistent_newbest_pending;
    trigger_db = false;

    % =================================================================
    % [R4.11b] Adaptive stagnation detection using (1-alpha)/Dim
    % =================================================================
    if abs(f_best - prev_f_best) < stagnation_threshold
        stagnation_count = stagnation_count + 1;
    else
        stagnation_count = 0;
    end
    prev_f_best = f_best;

    progress = fes / FEmax;
    % Eq(26) progress-bin index (corrected boundaries): bin b = floor(progress*20) means the b-th
    % 5% checkpoint has been crossed (b=1..19); no clamping, the 1<=current_bin<=19 gate below decides triggering.
    current_bin = floor(progress * 20);

    if current_bin >= 1 && current_bin <= 19
        bin_idx = current_bin;

        if ~db_triggered_bins(bin_idx) && f_second_best ~= inf && fes < FEmax

            if Dim < dim_low_threshold
                if has_reinforce_right
                    trigger_db = true;
                end
            elseif Dim <= dim_mid_threshold && Dim >= dim_low_threshold
                if stagnation_count >= stagnation_thres || ...
                        (has_reinforce_right && rand() > prob_new_best)
                    trigger_db = true;
                end
            else
                if stagnation_count >= stagnation_thres
                    trigger_db = true;
                end
            end

            if trigger_db
                % ==========================================================
                % [R4.11a] Record Hamming distance before AECO trigger
                % ==========================================================
                b_best   = double(x_best > thres);
                b_second = double(x_second_best > thres);
                h_dist   = sum(b_best ~= b_second);

                hamming_log.trigger_count = hamming_log.trigger_count + 1;
                hamming_log.distances(end+1) = h_dist;
                hamming_log.fes_at_trigger(end+1) = fes;

                [f_best, e_best, x_best, curve_hist, ErrorCurve, Nf_hist, fes, db_diag] = ...
                    run_db_and_update(f_best, e_best, x_best, ...
                    curve_hist, ErrorCurve, Nf_hist, fes, FEmax, ...
                    HO, thres, fitnessfun, feature, label, x_second_best);

                % [R4.11a] Record AERS duplicate/unique subset diagnostics
                db_diag_log.trigger_count = db_diag_log.trigger_count + 1;
                db_diag_log.total_evaluated(end+1)  = db_diag.total_evaluated;
                db_diag_log.unique_evaluated(end+1) = db_diag.unique_evaluated;
                db_diag_log.duplicate_count(end+1)  = db_diag.duplicate_count;
                db_diag_log.fes_at_trigger(end+1)   = fes;

                db_triggered_bins(bin_idx) = true;
                persistent_newbest_pending = false;
            end

        end
    end

    if ~trigger_db && new_best_found
        persistent_newbest_pending = true;
    end
end

% ----------------- Post-processing -----------------
if fes < FEmax
    curve_hist((fes+1):FEmax) = f_best;
    ErrorCurve((fes+1):FEmax) = e_best;
    Nf_hist((fes+1):FEmax) = numel(find(x_best > thres));
end

Select_Index = find(x_best > thres);
Sf    = Select_Index;
Nf    = length(Sf);
sfeature = feature(:, Sf);
curve = curve_hist;
NfCurve = Nf_hist;

% =================================================================
% [R4.10 / R4.11a] Package all diagnostics for output
% =================================================================
% Trim ADES log to actual number of iterations
ades_log.iter          = ades_log.iter(1:iter);
ades_log.fes           = ades_log.fes(1:iter);
ades_log.ExploreSize   = ades_log.ExploreSize(1:iter);
ades_log.DevelopSize   = ades_log.DevelopSize(1:iter);
ades_log.progress_frac = ades_log.progress_frac(1:iter);

diagnostics.ades_log        = ades_log;
diagnostics.hamming_log     = hamming_log;
diagnostics.db_diag_log     = db_diag_log;
diagnostics.total_iters     = iter;
diagnostics.total_fes       = fes;
diagnostics.stagnation_threshold = stagnation_threshold;
diagnostics.final_ExploreSize    = ExploreSize;
diagnostics.final_DevelopSize    = DevelopSize;

% ---- Nested function: population evaluation ----
    function [fit_res, err_res] = evaluatePopulation(population, popSize)
        fit_array = zeros(popSize, 1);
        Err_array = zeros(popSize, 1);
        for i = 1:popSize
            if fes >= FEmax, break; end
            [fit_array(i), Err_array(i)] = fitnessfun(feature, label, population(i,:), HO, thres, 'KNN');
            fes = fes + 1;

            if fit_array(i) < f_best
                f_second_best = f_best;
                x_second_best = x_best;
                f_best = fit_array(i);
                e_best = Err_array(i);
                x_best = population(i,:);
            elseif fit_array(i) >= f_best && fit_array(i) < f_second_best
                f_second_best = fit_array(i);
                x_second_best = population(i,:);
            end

            if fes <= length(curve_hist)
                curve_hist(fes) = f_best;
                ErrorCurve(fes) = e_best;
                Nf_hist(fes) = numel(find(x_best > thres));
            end
        end
        fit_res = fit_array;
        err_res = Err_array;
    end
end


% ---- DB (AERS) ----
function [f_best, e_best, x_best, curve_hist, ErrorCurve, Nf_hist, fes, db_diag] ...
    = run_db_and_update(f_best, e_best, x_best, curve_hist, ErrorCurve, Nf_hist, fes, FEmax, HO, thres, fitnessfun, feature, label, x_second_best)

runDB_DimensionSelection = @tool_alogrithm.DB.runDB_DimensionSelection;
[db_solution, db_fitness, db_err, db_fes, db_diag] = runDB_DimensionSelection(...
    x_best, x_second_best, feature, label, HO, thres, fitnessfun);

start_idx = fes + 1;
end_idx = min(FEmax, fes + db_fes);

% Behavior when no mask improvement (Reviewer 1 item 4): if the AERS best mask fitness is not better than
% the current incumbent (db_fitness >= f_best), keep the original incumbent unchanged and only consume fes.
if db_fitness < f_best
    f_best = db_fitness;
    e_best = db_err;
    x_best = db_solution;
end

if start_idx <= FEmax && end_idx >= start_idx
    curve_hist(start_idx:end_idx) = f_best;
    ErrorCurve(start_idx:end_idx) = e_best;
    Nf_hist(start_idx:end_idx) = numel(find(x_best > thres));
end

fes = fes + db_fes;
end
