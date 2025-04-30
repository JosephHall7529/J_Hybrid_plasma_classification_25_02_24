function plot_layout(ℓ)
    @assert ℓ <= 10
    if ℓ <= 3
        layout = [(j, 1) for j in 1:ℓ]
        □ = (700, 500*ℓ)
    elseif ℓ == 5
        layout = [(j, 1) for j in 1:ℓ]
        □ = (700, 500*ℓ)
    elseif mod(ℓ, 2) == 0
        layout = [(i, j) for (i, j) in Iterators.product(1:div(ℓ, 2), 1:2)]
        □ = (1400, 500*div(ℓ, 2))
    else
        layout = vcat([(j, 1) for j in 1:div(ℓ, 2)+1], [(j, 2) for j in 1:div(ℓ, 2)])
        □ = (1400, 500*(div(ℓ, 2)+1))
    end
    return layout, □
end

# plotting variability of regression parameters
function regression_viability(D::DataFrame, dependent::String, predictors::Vector{String}; kwargs...)
    parameters = vcat(dependent, predictors)
    ℓ = length(parameters)
    layout, □ = plot_layout(ℓ)

    fig = Figure(size=□)

    axes = []
    for (k, sub_) in enumerate(layout)
        ax = Axis(fig[sub_...]; kwargs...)
        hist!(ax, D[:, parameters[k]], normalization=:probability)
        ax.xlabel = LaTeX_dict[parameters[k]]
        ax.ylabel = "P(X=x)"
        push!(axes, ax)
    end
    fig
end

# regression_viability(AUG, "TAUTH", ["IP", "BT", "NEL", "PLTH"])

# plotting time traces
function plot_1D_features(ts::Tuple{String, Number}, feat::String; kwargs...)
    P = profiles(ts..., feat)  

    fig, ax, line = lines(P.t, abs.(P.y[:, 1].*normalise_2D_features[feat]); kwargs...)
    ax.xlabel = "t"
    ax.ylabel = feat
    ax.title = "$ts"
    # axislegend(ax)
    return fig
end
function plot_1D_features(ts::Tuple{String, Number}, feats::Vector{String}; kwargs...)
    ℓ = length(feats)
    layout, □ = plot_layout(ℓ)

    P = [profiles(ts..., feat) for feat in feats]  
    Y = [abs.(P[n].y.y.*normalise_2D_features[feat]) for (n, feat) in enumerate(feats)]

    fig = Figure(size=□);

    axes = []
    for (k, sub_) in enumerate(layout)
        ax = Axis(fig[sub_...]; kwargs...)
        lines!(ax, P[k].t, Y[k])
        ax.xlabel = "t"
        ax.ylabel = feats[k]
        ax.title = "$ts"
        push!(axes, ax)
    end

    return axes, fig
end
function plot_1D_features(ts::Tuple{String, Number}, feat::String, shade::Bool; kwargs...) 
    if !shade
        return plot_1D_features(ts, feat)
    end 

    P = profiles(ts..., feat)
    Y = abs.(P.y.y .* normalise_2D_features[feat])

    t = Dict("IP" => FTIP[ts], "NBI" => FTNBI[ts])
    BV = Dict("IP" => (t["IP"][1] .< P.t .< t["IP"][2]), "NBI" => (t["NBI"][1] .< P.t .< t["NBI"][2]))

    x_shade = Dict("IP" => P.t[BV["IP"]], "NBI" => P.t[BV["NBI"]])
    y_shade = Dict("IP" => Y[BV["IP"]], "NBI" => Y[BV["NBI"]])

    fig = Figure();
    ax = Axis(fig[1, 1]; kwargs...)
    lines!(ax, P.t, Y)
    band!(ax, x_shade["IP"], 0, y_shade["IP"], color=(:red, 0.2), label="flat top IP")
    band!(ax, x_shade["NBI"], 0, y_shade["NBI"], color=(:blue, 0.2), label="flat top NBI")

    ax.xlabel = "t"
    ax.ylabel = feat
    ax.title = "$ts"
    axislegend(ax, position=:lt)
    return fig
end
function plot_1D_features(ts::Tuple{String, Number}, feats::Vector{String}, shade::Bool; kwargs...)
    if !shade
        return plot_1D_features(ts, feats)
    end 
    
    ℓ = length(feats)
    layout, □ = plot_layout(ℓ) 

    P = [profiles(ts..., feat) for feat in feats]  
    Y = [abs.(P[n].y.y.*normalise_2D_features[feat]) for (n, feat) in enumerate(feats)]

    t = Dict("IP" => FTIP[ts], "NBI" => FTNBI[ts])
    BV = Dict("IP" => [(t["IP"][1] .< P[k].t .< t["IP"][2]) for k in 1:ℓ], "NBI" => [(t["NBI"][1] .< P[k].t .< t["NBI"][2]) for k in 1:ℓ])
    
    x_shade = Dict("IP" => [P[k].t[BV["IP"][k]] for k in 1:ℓ], "NBI" => [P[k].t[BV["NBI"][k]] for k in 1:ℓ])
    y_shade = Dict("IP" => [Y[k][BV["IP"][k]] for k in 1:ℓ], "NBI" => [Y[k][BV["NBI"][k]] for k in 1:ℓ])

    fig = Figure(size=□);
    
    axes = []
    for (k, sub_) in enumerate(layout)
        ax = Axis(fig[sub_...]; kwargs...)
        lines!(ax, P[k].t, Y[k])
        band!(ax, x_shade["IP"][k], 0, y_shade["IP"][k], color=(:red, 0.2), label="90% IP")
        band!(ax, x_shade["NBI"][k], 0, y_shade["NBI"][k], color=(:blue, 0.2), label="80% NBI")
        ax.xlabel = "t"
        ax.ylabel = "$(feats[k])"
        push!(axes, ax)
    end
	axes[1].title = "$(ts[1]) #$(ts[2])"
    axislegend(axes[minimum([4, ℓ])], position=:lt)
    display(fig)
    return axes, fig
end

# begin
#     axes, fig = plot_1D_features(("aug", 20115), ["IP", "PNBI", "PECRH", "Q95", "BETAPOL"], false)
#     fig
# end
# using GLMakie
# GLMakie.activate!()
# axes, fig = plot_1D_features(("aug", 17220), features, limits=((0, 8), nothing))
# axes[4].limits = ((0, 8), (0, 1.3))
# # axes[3].limits = ((0, 4.0), nothing)

# plot_1D_features(("aug", 29185), ["IP", "PNBI"], true)