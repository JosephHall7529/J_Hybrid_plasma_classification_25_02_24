function original_space(data::hybrid_classification)
    return data.data_0D.data.original_space
end
function original_space(data::data0D)
    return data.data.original_space
end
function original_space(data::global_data)
    return data.original_space
end

function log_space(data::hybrid_classification)
    return data.data_0D.data.log_space
end
function log_space(data::data0D)
    return data.data.log_space
end
function log_space(data::global_data)
    return data.log_space
end

function baseline(data::hybrid_classification)
    return data.data_0D.baseline
end
function hybrid(data::hybrid_classification)
    return data.data_0D.hybrid
end
function tbd(data::hybrid_classification)
    return data.data_0D.tbd
end
function baseline(data::data0D)
    return data.baseline
end
function hybrid(data::data0D)
    return data.hybrid
end
function tbd(data::data0D)
    return data.tbd
end

function profiles()
    return DATA_.data_2D.data
end
function profiles(tok::String)
    return DATA_.data_2D.data[tok]
end
function profiles(tok::String, shot::Int64)
    return DATA_.data_2D.data[tok].profiles[shot]
end
function profiles(tok, shot::Int64, feat::String)
    profile_shot = profiles(tok, shot) 
    if typeof(profile_shot) == ITPA_profile_extract
        return profile_shot.data[feat]
    elseif typeof(profile_shot) == SAL_profile_extract
        return profile_shot.data[SAL_dict[feat]][feat]
    elseif typeof(profile_shot) == AUG_profile_extract
        return profile_shot.data[AUG_dict[feat]][feat]
    end 
end
function profiles(data::data2D)
    return data.data
end
function profiles(data::data2D, tok::String)
    return data.data[tok]
end
function profiles(data::data2D, tok::String, shot::Int64) 
    return data.data[tok].profiles[shot]
end
function profiles(data::data2D, tok, shot::Int64, feat::String)
    profile_shot = profiles(data, tok, shot) 
    if typeof(profile_shot) == ITPA_profile_extract
        return profile_shot.data[feat]
    elseif typeof(profile_shot) == SAL_profile_extract
        return profile_shot.data[SAL_dict[feat]][feat]
    elseif typeof(profile_shot) == AUG_profile_extract
        return profile_shot.data[AUG_dict[feat]][feat]
    end 
end

# profile data
function id!(data::data2D)
    for row in eachrow(id_codes_2D)
        tok = row.TOK
        shot = row.SHOT
        prof = profiles(data, tok, shot)
        prof.id = String((@subset id_codes_2D @byrow begin 
            :TOK == tok
            in(:SHOT, shot)
        end).id[1])
    end
end
function id!(data::hybrid_classification)
    id!(data.data_2D)
end

import Base.which
function which(feature::String, version::String=""; D::DataFrame=all_2D_features, count::Bool=false)
    if feature == "HYBRID"
        data = filter(Meta.parse("HYBRID_$(version)") => ==("YES"), D)
    elseif feature == "BASELINE"
        data = filter(Meta.parse("HYBRID_$(version)") => ==("NO"), D)
    else
        data = filter(Meta.parse(feature) => ==(1.0), D)
    end
    if !count
        return data
    else
        cnt = Dict(features_2D .=> [+(data[!, feat]...) for feat in features_2D])
        return data, cnt
    end
end
function which(features::Vector{String}, version::String=""; D::DataFrame=all_2D_features, count::Bool=false)
    data = D
    for feature in features
        data = which(feature, version; D=data)
    end
    if !count
        return data
    else
        cnt = Dict(features_2D .=> [+(data[!, feat]...) for feat in features_2D])
        return data, cnt
    end
end
function which(toks::Vector{String}, features::Union{String, Vector{String}}, version::String=""; D::DataFrame=all_2D_features, count::Bool=false) 
    D = (@subset D @byrow in(:TOK, toks))
    return which(features, version, D=D, count=count)
end
function which(tok::String, features::Union{String, Vector{String}}, version::String=""; D::DataFrame=all_2D_features, count::Bool=false) 
    D = (@subset D @byrow :TOK == tok)
    return which(features, version, D=D, count=count)
end

import Base.+
function +(variable_profiles::Vector{variable_profile}, new_feature::String)

    PROFS_ = deepcopy(variable_profiles)
    ℓ = length(variable_profiles)

    T = Vector{Vector{Float64}}(undef, ℓ)
    ρ_all = Vector{Vector{Float64}}(undef, ℓ)
    gbdf = Vector{GroupedDataFrame{DataFrame}}(undef, ℓ)

    for (k, prof) in enumerate(PROFS_)
        rename!(prof.y, ["t$(Int64(round(i * 1e6, digits=0)))" for i in prof.t])
        insertcols!(prof.y, 1, :ρ => prof.ρ) 

        T[k] = prof.t
        ρ_all[k] = prof.ρ  

        gbdf[k] = groupby(prof.y, :ρ) 
    end

    t_all = sort(unique(vcat(T...)))
    actionable_t = ["t$(Int64(round(t * 1e6, digits=0)))" for t in t_all]
    ρ_all = sort(unique(vcat(ρ_all...)))

    final = DataFrame()
    for ρi in ρ_all
        tmp = Vector{DataFrame}()
        for k in 1:ℓ
            if in(ρi, PROFS_[k].ρ)
                D = gbdf[k][(ρ=ρi,)]
                headers = names(D)
                push!(tmp, DataFrame(headers .=> mean.(eachcol(D))))
            end
        end
        dict = Dict(actionable_t .=> zeros(length(t_all)))
        dict["ρ"] = ρi
        for (t, t_label) in zip(t_all, actionable_t)
            for k in 1:ℓ
                if in(t, T[k])
                    dict["$t_label"] += tmp[k][1, "$t_label"]
                end
            end
        end
        append!(final, DataFrame(dict))
    end
    y = sort(select(final, actionable_t))
    rename!(y, actionable_t .=> ["x$i" for i in 1:length(t_all)])
    
    variable_profile(new_feature, sort(t_all), ρ_all, y, length(ρ_all), Vector{Tuple}(), DataFrame())
end

function tok_shots(feature::Union{String, Vector{String}}, version::String="")
    Tuple.(eachrow(which(feature, version)[!, [:tok, :shot]]))
end
function tok_shots(D::Union{DataFrame, SubDataFrame}) 
    Tuple.(eachrow(D[!, [:tok, :shot]]))
end

function find_2D(id::String; metadata=false)
    tok, shot = (@subset id_codes_2D @byrow :id == id)[1, [:TOK, :SHOT]]
    if !metadata
        profiles(tok, shot)
    elseif metadata
        DataFrame(:id => id, :TOK => tok, :SHOT => shot, :HYBRID => "")
    end
end

function find_0D(id::String; metadata=false) 
    D = original_space(DATA_) 
    if !metadata
        return @subset D @byrow :id == "$id"
    elseif metadata
        (@subset D @byrow :id == "$id")[!, [:id, :TOK, :SHOT, :HYBRID]]
    end
end

function classification_v1!(data_0D::data0D)
    df = data_0D.data.original_space
    class = df[!, [:id, :TOK, :SHOT, :HYBRID]]
    class.class_ID .= "original classification"
    class.comments .= ""
    data_0D.hybrid = @subset class @byrow :HYBRID == "YES"
    data_0D.baseline = @subset class @byrow :HYBRID == "NO"
    data_0D.tbd = @subset class @byrow :HYBRID == "UNKNOWN"
    return data_0D 
end
function classification_v1!(data_2D::data2D; permanant::DataFrame=all_2D_features)
    class = DataFrame()
    for (id, tok, shot, ITPA_data) in eachrow(id_codes_2D)
        if in(shot, vcat(AUG_baseline, AUG_ITER_baseline, JET_baseline))
            append!(class, DataFrame(:id => id, 
                                    :TOK => tok, 
                                    :SHOT => shot, 
                                    :HYBRID => "NO"))
            continue
        elseif in(shot, vcat(AUG_hybrid, JET_hybrid))
            append!(class, DataFrame(:id => id, 
                                    :TOK => tok, 
                                    :SHOT => shot, 
                                    :HYBRID => "YES"))
            continue
        elseif ITPA_data == 1
            D = find_0D(id, metadata=true)
            append!(class, DataFrame(D[1, :]))
            continue
        else
            append!(class, DataFrame(:id => id, 
                                    :TOK => tok, 
                                    :SHOT => shot, 
                                    :HYBRID => "UNKNOWN"))
            continue
        end
    end
    class.class_ID .= "original classification"
    class.comments .= ""

    data_2D.hybrid = @subset class @byrow :HYBRID == "YES"
    data_2D.baseline = @subset class @byrow :HYBRID == "NO"
    data_2D.tbd = @subset class @byrow :HYBRID == "UNKNOWN"
    
    permanant.HYBRID_v1 .= "UNKNOWN"
    @eachrow! permanant :HYBRID_v1 = in(:id, data_2D.hybrid.id) ? "YES" : :HYBRID_v1 
    @eachrow! permanant :HYBRID_v1 = in(:id, data_2D.baseline.id) ? "NO" : :HYBRID_v1 
end
function classification_v1!()
    classification_v1!(DATA_.data_0D)
    classification_v1!(DATA_.data_2D)
    DATA_
end

function classification_v1!(data_0D::data0D)
    df = data_0D.data.original_space
    class = df[!, [:id, :TOK, :SHOT, :HYBRID]]
    class.class_ID .= "original classification"
    class.comments .= ""
    data_0D.hybrid = @subset class @byrow :HYBRID == "YES"
    data_0D.baseline = @subset class @byrow :HYBRID == "NO"
    data_0D.tbd = @subset class @byrow :HYBRID == "UNKNOWN"
    return data_0D 
end
function classification_v1!(data_2D::data2D; permanant::DataFrame=all_2D_features)
    class = DataFrame()
    for (id, tok, shot, ITPA_data) in eachrow(id_codes_2D)
        if in(shot, vcat(AUG_baseline, AUG_ITER_baseline, JET_baseline))
            append!(class, DataFrame(:id => id, 
                                    :TOK => tok, 
                                    :SHOT => shot, 
                                    :HYBRID => "NO"))
            continue
        elseif in(shot, vcat(AUG_hybrid, JET_hybrid))
            append!(class, DataFrame(:id => id, 
                                    :TOK => tok, 
                                    :SHOT => shot, 
                                    :HYBRID => "YES"))
            continue
        elseif ITPA_data == 1
            D = find_0D(id, metadata=true)
            append!(class, DataFrame(D[1, :]))
            continue
        else
            append!(class, DataFrame(:id => id, 
                                    :TOK => tok, 
                                    :SHOT => shot, 
                                    :HYBRID => "UNKNOWN"))
            continue
        end
    end
    class.class_ID .= "original classification"
    class.comments .= ""

    data_2D.hybrid = @subset class @byrow :HYBRID == "YES"
    data_2D.baseline = @subset class @byrow :HYBRID == "NO"
    data_2D.tbd = @subset class @byrow :HYBRID == "UNKNOWN"
    
    permanant.HYBRID_v1 .= "UNKNOWN"
    @eachrow! permanant :HYBRID_v1 = in(:id, data_2D.hybrid.id) ? "YES" : :HYBRID_v1 
    @eachrow! permanant :HYBRID_v1 = in(:id, data_2D.baseline.id) ? "NO" : :HYBRID_v1 
end
function classification_v1!()
    classification_v1!(DATA_.data_0D)
    classification_v1!(DATA_.data_2D)
    DATA_
end

function training_partion(labelled_data::OrderedDict, labels::Vector{String}, k::Int=2; S::Int=123)
    ℓ = countmap(labelled_data |> values)

    i = 0
    TD_ind = Vector{Int}(undef, 0)
    for label in labels
        TD_ind = vcat(TD_ind, sample(Random.seed!(S), i+1:i+ℓ[label], k, replace=false))
        i += ℓ[label]
    end
    
    return [(in(i, TD_ind) ? true : false) for i in 1:length(labelled_data)]
end
function training_partion(labelled_data::OrderedDict, labels::Vector{String}, k::Vector{Int}=[2]; S::Int=123)
    ℓ = countmap(labelled_data |> values)
    @assert length(labels) == length(k)

    i = 0
    TD_ind = Vector{Int}(undef, 0)
    for (label, ki) in zip(labels, k)
        TD_ind = vcat(TD_ind, sample(Random.seed!(S), i+1:i+ℓ[label], ki, replace=false))
        i += ℓ[label]
    end
    
    return [(in(i, TD_ind) ? true : false) for i in 1:length(labelled_data)]
end

function hyper_parameter_search(data::DTW_hyp_1, labelled_data::OrderedDict{Tuple{String, Int64}, String}, k::Union{Vector{Int}, Int}; interesting::String="", N::Int=5)
    labelled_ts = collect(labelled_data |> keys)
    labelled_y = collect(labelled_data |> values)
    shot_dict = Dict([a => n for (n, a) in enumerate(data.tok_shots)])
    labelled_ind = [shot_dict[k] for k in labelled_ts]

    cos_arr = Array(data.cosine_cost[:, 2:end])
    # mag_arr = Array(data.magnitude_cost[:, 2:end])
    ft_arr = Array(data.flat_top_cost[:, 2:end])
    # cos_med = quantile(cos_arr[labelled_ind, labelled_ind][:], start_quantile)
    # # mag_med = median(mag_arr[labelled_ind, labelled_ind][:])
    # ft_med = quantile(ft_arr[labelled_ind, labelled_ind][:], start_quantile)

	shot_dict = Dict([a => n for (n, a) in enumerate(data.tok_shots)])
    labels = collect(labelled_data |> values) |> unique

    cnt, ACC, δ_acc, cos_med, ft_med = 1, 0., 100, 0., 0.

    while (cnt < 8) && (ACC !== 1.) && (δ_acc > 0.002)
        if cnt == 1
            Cs = collect(Iterators.product( 
                quantile(cos_arr[labelled_ind, labelled_ind][:], [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95]),
                quantile(ft_arr[labelled_ind, labelled_ind][:], [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95])
            ))
        else
            Cs = collect(Iterators.product(
                range(cos_med/2, cos_med+(cos_med/2), length=3+(2*cnt)),
                # range(mag_med/2, mag_med*2, length=10),
                range(ft_med/2, ft_med+(ft_med/2), length=3+(2*cnt))
            ))
        end
        
        nc, nft = size(Cs)
        hyp_search = zeros(nc, nft, 3)
        acc_int = [zeros(nc, nft) for _ in 1:nthreads()]

        Threads.@threads for S in ProgressBar(1:N)
            # Random.seed!(S)
            train_ind = training_partion(labelled_data, labels, k, S=S)
            for (i, j) in Iterators.product(1:nc, 1:nft)
                CC = Cs[i, j][1]
                FTC = Cs[i, j][2]

                X = exp.(-(Array(data.cosine_cost[!, 2:end]) ./ CC).^2) .* 
                    exp.(-(Array(data.flat_top_cost[!, 2:end]) ./ FTC).^2)

                K = X[labelled_ind[train_ind], labelled_ind[train_ind]]
                model = svmtrain(K, labelled_y[train_ind], kernel=Kernel.Precomputed)

                KK = X[labelled_ind[train_ind], labelled_ind[Not(train_ind)]]
                ỹ, _ = svmpredict(model, KK)
                # ỹ, _ = KNN(KK, labelled_y[train_ind])
                acc_int[threadid()][i, j] += BalancedAccuracy()(ỹ, labelled_y[Not(train_ind)])
                hyp_search[i, j, 2] = CC
                hyp_search[i, j, 3] = FTC
            end
        end
        hyp_search[:, :, 1] = map(+, acc_int...)
        hyp_search[:, :, 1] ./= N
        # hyp_search |> display
    
        max = maximum(hyp_search[:, :, 1])
        inds = findall(i -> i == max, hyp_search[:, :, 1])
        ind = sample(Random.seed!(123), inds, 1)[1]

        max = round(max, digits=5)
        # last_cos_med, last_mag_med = cos_med, mag_med
        (cos_med, ft_med) = (hyp_search[ind, 2], hyp_search[ind, 3])
        println(max, ": ($cos_med, $ft_med)")
        δ_acc = abs(-(ACC, max))

        ACC = max 
        # no_good_values = length(inds)
        cnt += 1
    end
	X = exp.(-(Array(data.cosine_cost[!, 2:end]) ./ (cos_med)).^2) .* 
        exp.(-(Array(data.flat_top_cost[!, 2:end]) ./ (ft_med)).^2) 

    K = X[labelled_ind, labelled_ind]
    model = svmtrain(K, labelled_y, kernel=Kernel.Precomputed)

    KK = X[labelled_ind, Not(labelled_ind)]
    # ỹ, _ = KNN(KK, labelled_y)
    ỹ, confidence = svmpredict(model, KK)

    res = DataFrame(hcat(
            data.tok_shots[Not(labelled_ind)], ỹ
        ), [:ts, :predict]
    )
    if interesting !== ""
        show(stdout, "text/plain", (@subset res @byrow :predict == interesting))
    end
    return res, confidence, KK, (cos_med, ft_med), ACC
end 

function classify!(data::DTW_hyp_1, labelled_data::OrderedDict{Tuple{String, Int64}, String}, hyperparameters::Tuple; interesting::String="")

    labelled_ts = collect(labelled_data |> keys)
    labelled_y = collect(labelled_data |> values)
    shot_dict = Dict([a => n for (n, a) in enumerate(data.tok_shots)])
    labelled_ind = [shot_dict[k] for k in labelled_ts]

    X = exp.(-(Array(data.cosine_cost[!, 2:end]) ./ (hyperparameters[1])).^2) .* exp.(-(Array(data.flat_top_cost[!, 2:end]) ./ (hyperparameters[2])).^2) 

    K = X[labelled_ind, labelled_ind]
    model = svmtrain(K, labelled_y, kernel=Kernel.Precomputed, probability=true)

    KK = X[labelled_ind, Not(labelled_ind)]
    # ỹ, _ = KNN(KK, labelled_y)
    ỹ, confidence = svmpredict(model, KK)

    res = DataFrame(hcat(
            data.tok_shots[Not(labelled_ind)], ỹ
        ), [:ts, :predict]
    )
    if interesting !== ""
        show(stdout, "text/plain", (@subset res @byrow :predict == interesting))
    end
    return res, confidence, KK, model
end
