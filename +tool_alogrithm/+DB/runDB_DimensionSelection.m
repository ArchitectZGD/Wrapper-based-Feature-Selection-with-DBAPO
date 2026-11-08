function [best_sol_continuous, best_fit, best_err, fes_count, db_diag] = ...
    runDB_DimensionSelection(x_best, x_second_best, feature, label, HO, thres, fitnessfun)

    Dim = length(x_best);
    fes_count = 0;

    % --- DB parameters ---
    PopSize = 3;
    MaxGen = min(10, Dim);
    CrossoverRate = 0.8;
    MutationRate = 0.3;

    % [R4.11a] Track distinct candidates evaluated
    evaluated_candidates = {};  % store all candidate solutions as strings for uniqueness check

    % Initialise binary mask population
    B_pop = randi([0, 1], PopSize, Dim);
    B_fit = inf(PopSize, 1);
    B_err = inf(PopSize, 1);

    best_fit = inf;
    best_err = inf;
    best_chromosome = zeros(1, Dim);

    for gen = 1:MaxGen
        if fes_count >= PopSize * MaxGen, break; end

        for i = 1:PopSize
            continuous_sol = x_second_best;
            update_indices = (B_pop(i, :) == 1);
            continuous_sol(update_indices) = x_best(update_indices);

            [B_fit(i), B_err(i)] = fitnessfun(feature, label, continuous_sol, HO, thres);
            fes_count = fes_count + 1;

            % [R4.11a] Record this mask as a string for uniqueness counting
            evaluated_candidates{end+1} = char('0' + B_pop(i,:));
        end

        % Update best
        [min_fit_gen, min_idx_gen] = min(B_fit);
        if min_fit_gen < best_fit
            best_fit = min_fit_gen;
            best_err = B_err(min_idx_gen);
            best_chromosome = B_pop(min_idx_gen, :);
        end

        new_pop = zeros(size(B_pop));
        for k = 1:PopSize
            % Tournament selection
            p1_idx = randi(PopSize);
            p2_idx = randi(PopSize);
            if B_fit(p1_idx) < B_fit(p2_idx)
                parent1 = B_pop(p1_idx, :);
            else
                parent1 = B_pop(p2_idx, :);
            end

            p3_idx = randi(PopSize);
            p4_idx = randi(PopSize);
            if B_fit(p3_idx) < B_fit(p4_idx)
                parent2 = B_pop(p3_idx, :);
            else
                parent2 = B_pop(p4_idx, :);
            end

            % Crossover
            child = parent1;
            if rand < CrossoverRate
                crossover_point = randi(Dim - 1);
                child(crossover_point+1:end) = parent2(crossover_point+1:end);
            end

            % Asymmetric mutation
            for j = 1:Dim
                if rand < MutationRate
                    if child(j) == 1
                        if rand < 0.75
                            child(j) = 0;
                        end
                    else
                        if rand < 0.25
                            child(j) = 1;
                        end
                    end
                end
            end
            new_pop(k, :) = child;
        end

        B_pop = new_pop;
        % Elite preservation
        B_pop(1, :) = best_chromosome;
    end

    % Construct final solution from best mask
    best_sol_continuous = x_second_best;   % [FIX] use x_second_best as the base solution, consistent with evaluation (line 29), avoiding an all-zero solution that yields an empty Sf
    update_indices = (best_chromosome == 1);
    best_sol_continuous(update_indices) = x_best(update_indices);

    % =====================================================================
    % [R4.11a] AERS invocation diagnostics
    % =====================================================================
    n_total = numel(evaluated_candidates);
    n_unique = numel(unique(evaluated_candidates));
    db_diag.total_evaluated   = n_total;
    db_diag.unique_evaluated  = n_unique;
    db_diag.duplicate_count   = n_total - n_unique;
    db_diag.best_fit_improved = (best_fit < inf);  % did any mask beat infinity
end