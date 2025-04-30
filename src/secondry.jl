function emp(str::String)
    return eval(Meta.parse(str))
end

function indexstring!(D::DataFrame; override=false, colname="id", ℓ=6)
    if !in(colname, names(D)) || override
        gbdf = groupby(D, unique_comparison_parameters)
        for (n, df) in enumerate(gbdf)
            Random.seed!(n)
            df[!, colname] .= randstring(['A':'Z'; '0':'9'], ℓ)
        end
        D = vcat(gbdf...)
    end
    return select!(D, colname, :)
end

function index!(df::DataFrame; override=false, start=1)
    if !in("ind", names(df)) || override
        df[!, :ind] .= start:(start+size(df, 1)-1)
    end
    return select!(df, :ind, :)
end

function integrate_col(tok_shot::Tuple{String, Int}, feat::String)
    P = profiles(tok_shot..., feat)
    ℓ = length(P.t)
    diff = [P.t[i+1] - P.t[i] for i in 1:ℓ-1]
    y = abs.(P.y.y * normalise_2D_features[feat])
    cummulative = zeros(ℓ)
    for i in 2:ℓ
        cummulative[i] = (diff[i-1] * y[i-1]) + cummulative[i-1]
    end
    area_time = OrderedDict(cummulative .=> P.t)
    area_time
end
function integrate_row(tok_shot::Tuple{String, Int}, feat::String, ℓ=100; y_cutoff::Bool=false)
    P = profiles(tok_shot..., feat)
    y = abs.(P.y.y * normalise_2D_features[feat])
    y_steps = range(extrema(y)..., length=ℓ)
    y_diff = Float64(y_steps.step)
    y_steps = collect(y_steps)
    x_ranges = [(P.t[findfirst(i -> i >= y_steps[k], y)], P.t[findlast(i -> i >= y_steps[k], y)]) for k in 1:ℓ]

    cummulative = zeros(ℓ)
    for i in 2:ℓ
        cummulative[i] = (y_diff * abs(-(x_ranges[i]...))) + cummulative[i-1]
    end

    area_time = OrderedDict(cummulative .=> x_ranges)
    if !y_cutoff
        area_time
    else
        y_steps, area_time 
    end
end

function time_interval_top_down(tok_shot::Tuple{String, Int}, feat::String, percentage::Float64=0.75; rtol::Float64=0.1, divisions::Int=3, n_max::Int=200)
    P = profiles(tok_shot..., feat)
    y = abs.(P.y.y * normalise_2D_features[feat])
    ℓ = length(P.t)
    Pt = P.t
    
    if ℓ >= n_max
        step = div(ℓ, n_max)
        y = y[1:step:end]
        Pt = P.t[1:step:end]
        ℓ = length(y)
    end

    t_diff = vcat([Pt[i+1] - Pt[i] for i in 1:ℓ-1], 0.)
    N, full_int, cutoff, err, y_min, y_max, times, ind = 0, 0.0, 0.0, 1.0, minimum(y), maximum(y), Vector{Tuple}(), 0
    
    while (N < 30) && (err > rtol)
        y_steps = collect(range(y_max, y_min, length=divisions))
        times = [(Pt[findfirst(i -> i >= yi, y)], Pt[findlast(i -> i >= yi, y)]) for yi in y_steps]

        cum = zeros(divisions)
        for (n, t) in enumerate(times)
            BV = t[1] .< Pt .< t[2]
            cum[n] = dot(t_diff[BV], y[BV])
        end

        if N == 0
            full_int = cum[end]
            cutoff = full_int * percentage
        end

        ind = findfirst(i -> i > cutoff, cum)
        if ind == 1
            break
        end
        
        y_min, y_max = y_steps[ind], y_steps[ind-1]
        err = abs(-(cum[ind], cutoff))

        N += 1
    end
    return times[ind]
end

function flat_top_IP(tok_shots::Vector{Tuple{String, Int64}}, percentage::Float64=0.9) 
    dict = Dict{Tuple, Tuple}()
    for ts in tok_shots
        dict[ts] = time_interval_top_down(ts, "IP", percentage)
    end

    return dict
end
function flat_top_NBI(tok_shots::Vector{Tuple{String, Int64}}, percentage::Float64=0.8) 
    dict = Dict{Tuple, Tuple}()
    for ts in tok_shots
        dict[ts] = time_interval_top_down(ts, "PNBI", percentage)
    end

    return dict
end

function df_ts_naming(tok_shotz::Vector{Tuple{String, Int}})
    ["x_$(ts[1])_$(ts[2])" for ts in tok_shotz]
end

function least_data_length(tok_shotz::Vector{Tuple{String, Int}}, features::Vector{String})
    dict = Dict{Tuple{String, Int}, Dict{String, NamedTuple}}() 
    for ts in tok_shotz
        IP_t_size = [sum(FTIP[ts][1] .< profiles(ts..., feat).t .< FTIP[ts][2]) for feat in features]
        IP_smallest_size = minimum(IP_t_size)
        NBI_t_size = [sum(FTNBI[ts][1] .< profiles(ts..., feat).t .< FTNBI[ts][2]) for feat in features]
        NBI_smallest_size = minimum(NBI_t_size)
        dict[ts] = Dict("IP" => (feat = features[IP_t_size .== IP_smallest_size][1], length = IP_smallest_size),
                        "NBI" => (feat = features[NBI_t_size .== NBI_smallest_size][1], length = NBI_smallest_size)
        )
    end
    return dict
end

function write_table(model::StatsModels.TableRegressionModel; add::String="", kwargs...)
	if !in(:sigdigits, collect(keys(kwargs)))
		default = (; digits=3)
		kwargs = merge(default, kwargs)
	end
	coeffs = vcat(round(exp((model |> StatsBase.coef)[1]); kwargs...), round.((model |> StatsBase.coef)[2:end]; kwargs...))
	str = DataFrame(add.*regression_coefficients .=> coeffs)
	return str
end
function write_table(model::Array; parameters=regression_coefficients, add::String="", α0::String="exp", kwargs...)	
	if !in(:sigdigits, collect(keys(kwargs)))
		default = (; digits=3)
		kwargs = merge(default, kwargs)
	end
	if α0 == ""
		coeffs = vcat(round(model[1]; kwargs...), round.(model[2:end]; kwargs...))
	else
		coeffs = vcat(round(emp(α0)(model[1]); kwargs...), round.(model[2:end]; kwargs...))
	end
	str = DataFrame(add.*parameters .=> coeffs)
	return str
end

function stratified_k_fold(D::DataFrame, target::Symbol, k::Int)
    
    gbdf = groupby(D, :current_heating)
    for df in gbdf
        df.fold = shuffle!((1:nrow(df)) .% 5) 
    end
    D = vcat(gbdf...)

    get_fold_data(df, fold) =
        (train = view(df, df.fold .!= fold, :),
        test = view(df, df.fold .== fold, :)) 
    
    [get_fold_data(D, fold) for fold in 1:k]
end
