### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 488ce488-03e6-11f0-1466-c9a64e167aec
begin
	using Pkg
	Pkg.activate("/Users/joe/PHDProject/Thesis/code/src")
	Pkg.instantiate()
	using PlutoUI,
	CairoMakie,
	DynamicAxisWarping,
	Distances,
	Statistics,
	LIBSVM,
	Random,
	FileIO,
	MLJBase,
	DataFrames,
	StatsBase
end

# ╔═╡ b4e87f16-7061-4d41-8cde-d592d7899f66
using OrderedCollections,
	ThreadTools,
	DataFramesMeta

# ╔═╡ b33afb7b-a9a4-4f60-8719-1e1843825801
using ColorSchemes

# ╔═╡ 7de65134-7137-412d-96dc-9c84bfb5fb3c
include("/Users/joe/Project/Coding_clean/J_Hybrid_plasma_classification_25_02_24/src/main.jl")

# ╔═╡ 7c89f02a-7441-4cba-913d-35ce7d45ace9
md"# Dynamic time warping"

# ╔═╡ dc5be29b-9eb3-4877-bb61-d9766172f585
md"We take two time series signals:"

# ╔═╡ 2cd36f54-c0c6-4f8b-821a-19f2614a8475
# begin
# 	signal_1 = [0, 0, 0, 0, 1, 1, 0, 0, 0, -2, -1, 0, 0]
# 	signal_2 = [0, 0, 1, 1, 0, 0, -2, 0, 0, 0, 0]
# 	mat = dtw_cost_matrix(signal_2, signal_1, Cityblock())
# 	cost, i1, i2 = DynamicAxisWarping.trackback(mat)
# end

# ╔═╡ bed62eae-90d7-4660-9e69-6121c866b0fc
# let
# 	fig, ax, lin = lines(signal_1 .+ 0.6, label="signal_1")
# 	lines!(ax, signal_2 .- 0.6, label="signal_2")
# 	ax.xticks=0:13
# 	ax.yticklabelsvisible=false
# 	ax.xlabel="time"
# 	axislegend(ax)
# 	ax.xgridvisible = false
# 	ax.ygridvisible = false
# 	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/DTW_two_signals.png", fig)
# 	fig
# end;

# ╔═╡ 6ced956a-38ac-4a86-9b23-1c469e8b3a95
# let
# 	fig = Figure();
# 	ax = Axis(fig[1:11, 3:14])
# 	heatmap!(mat, colormap=:thermal)
# 	# contour!(mat; color = :white, levels=1:7, labels=true, labelfont=:bold, labelsize = 12)
# 	lines!(ax, i2, i1, color=:white, linewidth=3)
# 	ax.xticklabelsvisible=false
# 	ax.yticklabelsvisible=false
# 	ax.xticks=1:13
# 	ax.xticksvisible=false
# 	ax.yticksvisible=false

# 	# text!((2, 4), text="hi")
# 	for (i, j) in Iterators.product(1:13, 1:11)
# 		text!((i-0.1, j-0.3),
# 		    text="$(mat[i, j])",
# 		    color=(:white, 0.5)
# 		)
# 	end
# 	ax1 = Axis(fig[1:11, 1:2])
# 	lines!(ax1, -signal_2, 1:11, color=:orange)
# 	ax1.topspinevisible=false
# 	ax1.leftspinevisible=false
# 	ax1.rightspinevisible=false
# 	ax1.bottomspinevisible=false
# 	ax1.xgridvisible=false
# 	ax1.ygridvisible=false
# 	ax1.yticklabelsvisible=false
# 	ax1.xticklabelsvisible=false
# 	ax1.yticksvisible=false
# 	ax1.xticksvisible=false

# 	ax2 = Axis(fig[12:14, 3:14])
# 	lines!(ax2, signal_1, color=:blue)
# 	ax2.topspinevisible=false
# 	ax2.leftspinevisible=false
# 	ax2.rightspinevisible=false
# 	ax2.bottomspinevisible=false
# 	ax2.xgridvisible=false
# 	ax2.ygridvisible=false
# 	ax2.yticklabelsvisible=false
# 	ax2.xticklabelsvisible=false
# 	ax2.yticksvisible=false
# 	ax2.xticksvisible=false

	
# 	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/DTW_cost_matrix.png", fig)
# 	fig
# end;

# ╔═╡ 6a5bf655-ae06-4ef4-b44b-d0d55d729198
# let
# 	znorm(x) = (x = x.- mean(x); x ./= std(x))

# 	separation=0.7
# 	ds=1
# 	x, y = (signal_2, signal_1)
#     s1 = x .- separation
#     s2 = y .+ separation
# 	i = fill(Inf, 1, length(i1))
# 	p1, p2 = vec([i1'; i2'; i][:, 1:ds:end]), vec([s1[i1]'; s2[i2]'; i][:, 1:ds:end])

# 	fig, ax, lin = lines(s2)
# 	lines!(s1)
# 	lines!(p1, p2, color=(:gray, 0.5))
# 	ax.xticks=1:13
# 	ax.yticklabelsvisible=false
# 	ax.xticklabelsvisible=false
# 	ax.xgridvisible = false
# 	ax.ygridvisible = false

# 	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/DTW_matchplot.png", fig)
# 	fig
# end;

# ╔═╡ e00cc312-3368-4826-a48a-ecdd698d88ce
# let
# 	fig = Figure()
# 	ax1 = Axis(fig[1, 1])
# 	ax1.topspinevisible=false
# 	ax1.leftspinevisible=false
# 	ax1.rightspinevisible=false
# 	ax1.bottomspinevisible=false
# 	ax1.xgridvisible=false
# 	ax1.ygridvisible=false
# 	ax1.yticklabelsvisible=false
# 	ax1.xticklabelsvisible=false
# 	ax1.yticksvisible=false
# 	ax1.xticksvisible=false
# 	ax2 = Axis(fig[1, 2])
# 	ax2.topspinevisible=false
# 	ax2.leftspinevisible=false
# 	ax2.rightspinevisible=false
# 	ax2.bottomspinevisible=false
# 	ax2.xgridvisible=false
# 	ax2.ygridvisible=false
# 	ax2.yticklabelsvisible=false
# 	ax2.xticklabelsvisible=false
# 	ax2.yticksvisible=false
# 	ax2.xticksvisible=false
# 	ax3 = Axis(fig[2, 1:2])
# 	znorm(x) = (x = x.- mean(x); x ./= std(x))

# 	separation=0.7
# 	ds=1
# 	x, y = (signal_2, signal_1)
#     s1 = x .- separation
#     s2 = y .+ separation
# 	i = fill(Inf, 1, length(i1))
# 	p1, p2 = vec([i1'; i2'; i][:, 1:ds:end]), vec([s1[i1]'; s2[i2]'; i][:, 1:ds:end])

# 	lines!(ax3, s2)
# 	lines!(ax3, s1)
# 	lines!(ax3, p1, p2, color=(:gray, 0.5))
# 	ax3.xticks=1:13
# 	ax3.yticklabelsvisible=false
# 	ax3.xticklabelsvisible=false
# 	ax3.topspinevisible=false
# 	ax3.leftspinevisible=false
# 	ax3.rightspinevisible=false
# 	ax3.bottomspinevisible=false
# 	ax3.xgridvisible=false
# 	ax3.ygridvisible=false
# 	ax3.yticklabelsvisible=false
# 	ax3.xticklabelsvisible=false
# 	ax3.yticksvisible=false
# 	ax3.xticksvisible=false
# 	image!(ax1, rotr90(FileIO.load("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/DTW_two_signals.png")))

# 	image!(ax2, rotr90(FileIO.load("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/DTW_cost_matrix.png")))

# 	image!(ax3, rotr90(FileIO.load("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/DTW_matchplot.png")))
	
# 	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/DTW_description.png", fig)
# 	fig
# end

# ╔═╡ 82ac976d-ef0e-4736-96be-fa1aad8ddaec
md"# shot characteristics"

# ╔═╡ 1cc123b1-7c38-4083-b4b4-b3577cf0644c
let
	# 36177, 29772, 13622, 34610, 17282, 15714, 19314, 32135, 34441, 32464, 33407
	shot = 34841 
	x1_lim = 8.5
	x2_lim = 4
	fig1 = Figure(size=(1000, 400))
	ax1 = Axis(fig1[1:3, 2:4], title="#$shot", titlesize=15)
	ax2 = Axis(fig1[4:6, 2:4])
	for (n, (feat, norm, axis, label)) in enumerate(zip(["IP", "PNBI", "PECRH", "PICRH", "Q95", "BETAPOL", "NGW", "LI"], [1e-6, 1e-7, 1e-7, 1e-7, 1e-1, 1, 1, 1], [ax1, ax1, ax1, ax1, ax2, ax2, ax2, ax2], [L"I_p \, (10^{-6} MA)", L"P_{\mathrm{NBI}} \, (10^{-7} MA)", L"P_{\mathrm{ECRH}} \, (10^{-7} MA)", L"P_{\mathrm{ICRH}} \, (10^{-7} MA)", L"q_{95} \, (10^{-1})", L"\beta_p", L"f_{\mathrm{GW}}", L"\ell_i"]))
		P = profiles(("aug", shot)..., feat)
		lines!(axis, P.t, abs.(P.y.y .* norm), linewidth=1.5, colormap=:dracula, colorrange=(1, 8), color=n, label=label)
	end
	ax1.limits = ((0., x1_lim), (-0.1, 2.5))
	ax2.limits = ((0., x1_lim), (-0.1, 2.5))
	ax1.yticks = 0:0.5:2.5
	ax2.yticks = 0:0.5:2.5
	fig1[1:2, 1] = Legend(fig1, ax1, "", framevisible=false, labelsize=15)
	fig1[4:5, 1] = Legend(fig1, ax2, "", framevisible=false, labelsize=15)
	
	shot = 34770
	ax3 = Axis(fig1[1:3, 5:7], title="#$shot", titlesize=15)
	ax4 = Axis(fig1[4:6, 5:7])
	for (n, (feat, norm, axis, label)) in enumerate(zip(["IP", "PNBI", "PECRH", "PICRH", "Q95", "BETAPOL", "NGW", "LI"], [1e-6, 1e-7, 1e-7, 1e-7, 1e-1, 1, 1, 1], [ax3, ax3, ax3, ax3, ax4, ax4, ax4, ax4], [L"I_p \, (10^{-6} MA)", L"P_{\mathrm{NBI}} \, (10^{-7} MA)", L"P_{\mathrm{ECRH}} \, (10^{-7} MA)", L"P_{\mathrm{ICRH}} \, (10^{-7} MA)", L"q_{95} \, (10^{-1})", L"\beta_p", L"f_{\mathrm{GW}}", L"\ell_i"]))
		P = profiles(("aug", shot)..., feat)
		lines!(axis, P.t, abs.(P.y.y .* norm), linewidth=1.5, colormap=:dracula, colorrange=(1, 8), color=n, label=label)
	end
	ax3.limits = ((0., x2_lim), (-0.1, 2.5))
	ax4.limits = ((0., x2_lim), (-0.1, 2.5))
	ax3.yticks = 0:0.5:2.5
	ax4.yticks = 0:0.5:2.5
	# fig1[1:2, 8] = Legend(fig1, ax1, "", framevisible=false, labelsize=35)
	# fig1[4:5, 8] = Legend(fig1, ax2, "", framevisible=false, labelsize=35)

	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/shot_comparison.png", fig1)
	fig1
end

# ╔═╡ 54a0e9ca-2c72-4262-bd2c-8722b67aa8fa
md"## standardising DTW - IP"

# ╔═╡ fd21db59-a85e-4280-919c-2c77481d7c3e
let
	FTIP = flat_top_IP(tok_shots(which(["IP"])), 0.8)
	shot = 12021
	x1_lim = 8.6
	x2_lim = 4.5
	fig1 = Figure(size=(1900, 900))
	ax1 = Axis(fig1[1:3, 2:4], title="#$shot", titlesize=25)
	ax2 = Axis(fig1[4:6, 2:4])

	features = ["IP", "PNBI", "PECRH", "PICRH", "Q95", "BETAPOL", "NGW", "LI"]
	t = FTIP[("aug", shot)]
    
	for (n, (feat, norm, axis, label)) in enumerate(zip(features, [1e-6, 1e-7, 1e-7, 1e-7, 1e-1, 1, 1, 1], [ax1, ax1, ax1, ax1, ax2, ax2, ax2, ax2], [L"I_p \, (10^{-6} MA)", L"P_{\mathrm{NBI}} \, (10^{-7} MA)", L"P_{\mathrm{ECRH}} \, (10^{-7} MA)", L"P_{\mathrm{ICRH}} \, (10^{-7} MA)", L"q_{95} \, (10^{-1})", L"\beta_p", L"f_{\mathrm{GW}}", L"\ell_i"]))
		P = profiles(("aug", shot)..., feat)
		BV = t[1] .< P.t .< t[2]
		Y = abs.(P.y.y .* norm)
		x_shade = P.t[BV]
		y_shade = Y[BV]
		band!(axis, x_shade, 0, y_shade, color=(:cyan, 0.15))
		lines!(axis, P.t, abs.(P.y.y .* norm), linewidth=1.5, colormap=:dracula, colorrange=(1, 8), color=n, label=label)
	end
	ax1.limits = ((0, x1_lim), (-0.1, 2.5))
	ax2.limits = ((0, x1_lim), (-0.1, 2.5))
	ax1.yticks = 0:0.5:2.5
	ax2.yticks = 0:0.5:2.5
	fig1[1:2, 1] = Legend(fig1, ax1, "", framevisible=false, labelsize=35)
	fig1[4:5, 1] = Legend(fig1, ax2, "", framevisible=false, labelsize=35)
	
	shot = 34770
	t = FTIP[("aug", shot)]
	ax3 = Axis(fig1[1:3, 5:7], title="#$shot", titlesize=25)
	ax4 = Axis(fig1[4:6, 5:7])
	for (n, (feat, norm, axis, label)) in enumerate(zip(features, [1e-6, 1e-7, 1e-7, 1e-7, 1e-1, 1, 1, 1], [ax3, ax3, ax3, ax3, ax4, ax4, ax4, ax4], [L"I_p \, (10^{-6} MA)", L"P_{\mathrm{NBI}} \, (10^{-7} MA)", L"P_{\mathrm{ECRH}} \, (10^{-7} MA)", L"P_{\mathrm{ICRH}} \, (10^{-7} MA)", L"q_{95} \, (10^{-1})", L"\beta_p", L"f_{\mathrm{GW}}", L"\ell_i"]))
		P = profiles(("aug", shot)..., feat)
		BV = t[1] .< P.t .< t[2]
		Y = abs.(P.y.y .* norm)
		x_shade = P.t[BV]
		y_shade = Y[BV]
		band!(axis, x_shade, 0, y_shade, color=(:cyan, 0.15))
		lines!(axis, P.t, abs.(P.y.y .* norm), linewidth=1.5, colormap=:dracula, colorrange=(1, 8), color=n, label=label)
	end
	ax3.limits = ((0, x2_lim), (-0.1, 2.5))
	ax4.limits = ((0, x2_lim), (-0.1, 2.5))
	ax3.yticks = 0:0.5:2.5
	ax4.yticks = 0:0.5:2.5
	# fig1[1:2, 8] = Legend(fig1, ax1, "", framevisible=false, labelsize=35)
	# fig1[4:5, 8] = Legend(fig1, ax2, "", framevisible=false, labelsize=35)

	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/shot_comparison_stochastic_FTIP.png", fig1)
	fig1
end

# ╔═╡ a26ff402-ec87-4286-bdcd-d0159538d8db
let
	shot = 12021 
	x1_lim = 8.6
	x2_lim = 4.5
	fig1 = Figure(size=(1900, 900))
	ax1 = Axis(fig1[1:3, 2:4], title="#$shot", titlesize=25)
	ax2 = Axis(fig1[4:6, 2:4])

	features = ["IP", "PNBI", "PECRH", "PICRH", "Q95", "BETAPOL", "NGW", "LI"]
	t = FTNBI[("aug", shot)]
    
	for (n, (feat, norm, axis, label)) in enumerate(zip(features, [1e-6, 1e-7, 1e-7, 1e-7, 1e-1, 1, 1, 1], [ax1, ax1, ax1, ax1, ax2, ax2, ax2, ax2], [L"I_p \, (10^{-6} MA)", L"P_{\mathrm{NBI}} \, (10^{-7} MA)", L"P_{\mathrm{ECRH}} \, (10^{-7} MA)", L"P_{\mathrm{ICRH}} \, (10^{-7} MA)", L"q_{95} \, (10^{-1})", L"\beta_p", L"f_{\mathrm{GW}}", L"\ell_i"]))
		P = profiles(("aug", shot)..., feat)
		BV = t[1] .< P.t .< t[2]
		Y = abs.(P.y.y .* norm)
		x_shade = P.t[BV]
		y_shade = Y[BV]
		band!(axis, x_shade, 0, y_shade, color=(:indigo, 0.15))
		lines!(axis, P.t, abs.(P.y.y .* norm), linewidth=1.5, colormap=:dracula, colorrange=(1, 8), color=n, label=label)
	end
	ax1.limits = ((0, x1_lim), (-0.1, 2.5))
	ax2.limits = ((0, x1_lim), (-0.1, 2.5))
	ax1.yticks = 0:0.5:2.5
	ax2.yticks = 0:0.5:2.5
	fig1[1:2, 1] = Legend(fig1, ax1, "", framevisible=false, labelsize=35)
	fig1[4:5, 1] = Legend(fig1, ax2, "", framevisible=false, labelsize=35)
	
	shot = 34770
	t = FTNBI[("aug", shot)]
	ax3 = Axis(fig1[1:3, 5:7], title="#$shot", titlesize=25)
	ax4 = Axis(fig1[4:6, 5:7])
	for (n, (feat, norm, axis, label)) in enumerate(zip(features, [1e-6, 1e-7, 1e-7, 1e-7, 1e-1, 1, 1, 1], [ax3, ax3, ax3, ax3, ax4, ax4, ax4, ax4], [L"I_p \, (10^{-6} MA)", L"P_{\mathrm{NBI}} \, (10^{-7} MA)", L"P_{\mathrm{ECRH}} \, (10^{-7} MA)", L"P_{\mathrm{ICRH}} \, (10^{-7} MA)", L"q_{95} \, (10^{-1})", L"\beta_p", L"f_{\mathrm{GW}}", L"\ell_i"]))
		P = profiles(("aug", shot)..., feat)
		BV = t[1] .< P.t .< t[2]
		Y = abs.(P.y.y .* norm)
		x_shade = P.t[BV]
		y_shade = Y[BV]
		band!(axis, x_shade, 0, y_shade, color=(:indigo, 0.15))
		lines!(axis, P.t, abs.(P.y.y .* norm), linewidth=1.5, colormap=:dracula, colorrange=(1, 8), color=n, label=label)
	end
	ax3.limits = ((0, x2_lim), (-0.1, 2.5))
	ax4.limits = ((0, x2_lim), (-0.1, 2.5))
	ax3.yticks = 0:0.5:2.5
	ax4.yticks = 0:0.5:2.5
	# fig1[1:2, 8] = Legend(fig1, ax1, "", framevisible=false, labelsize=35)
	# fig1[4:5, 8] = Legend(fig1, ax2, "", framevisible=false, labelsize=35)

	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/shot_comparison_stochastic_FTNBI.png", fig1)
	fig1
end

# ╔═╡ 891e88be-17fd-4ab4-bcbd-e61312f56c93
begin
	features = ["IP", "PNBI"]
	tss = [("aug", shot) for shot in AUG_shots]
end

# ╔═╡ 1e205f99-ff30-4c90-b0eb-76e528819cab
let
	signal_1 = data_CEL.profile_data[("aug", 34774)]
	signal_2 = data_CEL.profile_data[("aug", 16788)]
	mat = dtw_cost_matrix(signal_2, signal_1, CosineDist(), transportcost=1.1)
	cost, i1, i2 = DynamicAxisWarping.trackback(mat)

	k=3
	fig = Figure();
	ax = Axis(fig[1:11, 3:14])
	heatmap!(mat, colormap=:thermal)
	# contour!(mat; color = :white, levels=1:7, labels=true, labelfont=:bold, labelsize = 12)
	lines!(ax, i2, i1, color=:white, linewidth=3)
	ax.xticklabelsvisible=false
	ax.yticklabelsvisible=false
	ax.xticks=1:13
	ax.xticksvisible=false
	ax.yticksvisible=false

	# text!((2, 4), text="hi")
	for (i, j) in Iterators.product(2:13:size(signal_1, 2), 1:13:size(signal_2, 2))
		text!((i-0.1, j-0.7),
		    text="$(round(mat[i, j], digits=2))",
		    color=(:white, 0.5)
		)
	end
	ax1 = Axis(fig[1:11, 1:2])
	lines!(ax1, -signal_2[1, :][:], 1:size(signal_2, 2), color=:darkorange)
	lines!(ax1, -signal_2[2, :][:]./10, 1:size(signal_2, 2), color=:orange, linestyle=(:dash, :dense))
	ax1.topspinevisible=false
	ax1.leftspinevisible=false
	ax1.rightspinevisible=false
	ax1.bottomspinevisible=false
	ax1.xgridvisible=false
	ax1.ygridvisible=false
	ax1.yticklabelsvisible=false
	ax1.xticklabelsvisible=false
	ax1.yticksvisible=false
	ax1.xticksvisible=false

	ax2 = Axis(fig[12:14, 3:14])
	lines!(ax2, 1:size(signal_1, 2), signal_1[1, :], color=:blue)
	lines!(ax2, 1:size(signal_1, 2), signal_1[2, :]./10, color=:skyblue, linestyle=(:dash, :dense))
	ax2.topspinevisible=false
	ax2.leftspinevisible=false
	ax2.rightspinevisible=false
	ax2.bottomspinevisible=false
	ax2.xgridvisible=false
	ax2.ygridvisible=false
	ax2.yticklabelsvisible=false
	ax2.xticklabelsvisible=false
	ax2.yticksvisible=false
	ax2.xticksvisible=false

	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/CO_EH_Cosine_dist.png", fig)
	fig
end

# ╔═╡ eb33a1b9-d8a0-4459-a1f8-7f3a22654d63
md"# Grid search algorithm"

# ╔═╡ c77f7e68-d4b7-4141-8260-f0b4eb14df19
md"# classifying early and late heating"

# ╔═╡ bf9d1595-2a00-4232-98ac-d4c73a850568
function current_heating(tok::String, shot::Int)
	int = (@subset all_2D_features @byrow begin
		:tok == tok
		:shot == shot
	end)
	return int.current_heating[1]
end

# ╔═╡ 208cf523-8ac5-403e-a835-5f6efac2544e
begin
    Random.seed!(17)
    CLn = 25
    CEn = 25
    COn = 5
    CLchoices = sort(sample(tok_shots((@subset all_2D_features @byrow in(:current_heating, ["C-LH"]))), CLn, replace=false))
    CEchoices = sort(sample(tok_shots((@subset all_2D_features @byrow in(:current_heating, ["C-EH"]))), CEn, replace=false))
    COchoices = sort(sample(tok_shots((@subset all_2D_features @byrow in(:current_heating, ["CO"]))), COn, replace=false))
end

# ╔═╡ c6b42a10-22ae-4a84-b265-fc3e0aa4b4fc
function cardinality_metadata(D::DataFrame, feature::Symbol; name::Symbol=:cardinality)
    no_of_points = DataFrame()
    gbdf, ind = grouping(D, feature)

    for id in ind
        ℓ = size(gbdf[id], 1)
        df = DataFrame(feature => id[1], name => ℓ)
        no_of_points = vcat(no_of_points, df)
    end
	
    df = DataFrame(feature => "Total", name => sum(no_of_points[!, name]))
    no_of_points = vcat(no_of_points, df)
    return no_of_points
end

# ╔═╡ aed01189-8018-43da-b275-0b39b84c396d
res, confidence, kernel_matrix, (cos_med, ft_med) = let
	labelled_data = let
	    labelled_shots = vcat(CLchoices, CEchoices, COchoices)
	    labels = vcat(repeat(["C-LH"], length(CLchoices)),
	        repeat(["C-EH"], length(CEchoices)),
	        repeat(["CO"], length(COchoices))) 
	    d = OrderedDict([ts for ts in labelled_shots] .=> labels)
	end
	hyper_parameter_search(data_CEL, labelled_data, [14,14,3], interesting="CO", N=200, start_quantile=0.4)
end

# ╔═╡ a70a39dc-9dbf-4739-aa92-f28d7aad6cc3
let
	shots = [j for (i, j) in res.ts]
	shot_dict = Dict([a => n for (n, a) in enumerate(data_BH.tok_shots)])

	labels = [current_heating("aug", shot) for shot in shots]
	
	res.label .= labels
	res.conf = Tuple.(zip(confidence[1, :], confidence[2, :]))
	curr_over = sort((@subset res @byrow :label == "CO"), :conf)

	acc, b_acc = Accuracy()(labels, res.predict), BalancedAccuracy()(labels, res.predict)
	println("accuracy = $(acc), balanced accuracy = $(b_acc)")

	col_dict = Dict("CO"=>1, "C-EH"=>2, "C-LH"=>3)
    ỹ = res.predict
    y = res.label
   
    f, a, s = scatter(confidence[1, :], confidence[2, :]; colormap=:tab10, 
        color=[col_dict[el] for el in ỹ],
		strokewidth = 0.1,
        label = [label => (;colormap=:tab10, colorrange=(1, 3), color=col_dict[label]) for label in ỹ],
		markersize = [(i == j ? 20 : 10) for (i, j) in zip(y, ỹ)],
        marker = [(i == j ? '∘' : (:utriangle)) for (i, j) in zip(y, ỹ)]
    )
	
    pos = [(:left, :bottom), (:left, :top), (:right, :top), (:right, :bottom)]
    for (n, (ts, coord)) in enumerate(zip(curr_over.ts, curr_over.conf))
        text!(a, coord, text="$(ts[2])", align=pos[mod(n+2,4)+1], fontsize=7)
    end

	# extra = (@subset res @byrow in(:ts, [("aug", shot) for shot in [27930]]))
	# for (n, (ts, coord)) in enumerate(zip(extra.ts, extra.conf))
 #        text!(a, coord, text="$(ts[2])", align=pos[mod(n,4)+1], fontsize=7)
 #    end
	
	a.xgridvisible = false
	a.ygridvisible = false
    axislegend(a, unique=true, merge=true, position=:lb)
    
	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/CO_EH_LH_classification.png", f)
	f
end

# ╔═╡ f162287a-6759-4fed-9767-1a86ab427f7b
let
	train_tss = vcat(CLchoices, CEchoices, COchoices)
	n = length(train_tss)
	ind = [data_CEL.shot_dict[ts] for ts in train_tss]

	fig = Figure();
	ax = Axis(fig[1, 1], xticks=(0.7:0.2:0.7, [L"\mathrm{DTW_{cos}}|_{I_p^{80%}}"]), xticklabelsize=20)
	ax1 = Axis(fig[1, 2], xticks=(0.7:1:1.7, [L"\mathrm{DTW_{mag}}|_{I_p^{80%}}", L"\mathrm{DTW_{mag}}|_{P_{NBI}^{70%}}"]), xticklabelsize=20)

	dat = data_CEL.cosine_cost
	mat = Array(dat[ind, ind.+1])
	CD = abs.(vcat([mat[i, i+1:n] for i in 1:n-1]...))
	hist!(ax, CD, scale_to=-0.6, offset=1, direction=:x, color=:gray70)
	hlines!(ax, [cos_med], xmin=0.05, xmax=0.95, color=:red)

	dat = data_CEL.flat_top_cost
	mat = Array(dat[ind, ind.+1])
	CD = abs.(vcat([mat[i, i+1:n] for i in 1:n-1]...))
	hist!(ax1, CD, scale_to=-0.6, offset=2, direction=:x, color=:gray50)
	hlines!(ax1, [ft_med], xmin=0.05, xmax=0.95, color=:red)
	
	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/cost_spread.png", fig)
	fig
end

# ╔═╡ 4f098996-b1d6-4c16-8068-0da5c2f7bdd7
let
	f = Figure(size=(1400, 800));
	a1 = Axis(f[1:3, 1:5], title="Late heating", ylabel="testing shots", ylabelsize=20, yticks=1:50:500, ylabelpadding=20)
	a2 = Axis(f[1:3, 6:10], yticklabelsvisible=false, yticksvisible=false, title="Early heating")
	a3 = Axis(f[1:3, 11], yticklabelsvisible=false, yticksvisible=false, title="Current overshoot")
	he = heatmap!(a1, kernel_matrix[1:25, :])
	heatmap!(a2, kernel_matrix[26:50, :])
	heatmap!(a3, kernel_matrix[51:55, :])
	Colorbar(f[1:3, 12], he, ticks=0:0.1:1, width=20)
	a1.xticks = (1:25, ["#$j" for (i,j) in CLchoices])
	a2.xticks = (1:25, ["#$j" for (i,j) in CEchoices])
	a3.xticks = (1:5, ["#$j" for (i,j) in COchoices])
	a1.xticklabelrotation = π/3
	a2.xticklabelrotation = π/3
	a3.xticklabelrotation = π/3
	a1.xticklabelsize = 15
	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/CEL_kernel_matrix.png", f)
	f
end

# ╔═╡ 4b08c5c7-f319-4544-91a0-00a2597dd64f
function KNN(kernel::Array, train_labels::Vector{String})
    tr_ℓ, te_ℓ = size(kernel)
    @assert length(train_labels) == tr_ℓ

    dict = Dict(1:tr_ℓ .=> train_labels)

    max_val = Vector{Number}(undef, te_ℓ)
    labels = Vector{Int}(undef, te_ℓ)
    for n in 1:te_ℓ
        max_val[n], labels[n] = findmax(kernel[:, n])
    end
    return [dict[element] for element in labels], max_val
end

# ╔═╡ e5763965-6c47-4e4b-97da-de0d63455a9c
let
	train_labels = vcat(repeat(["C-LH"], length(CLchoices)),
	        repeat(["C-EH"], length(CEchoices)),
	        repeat(["CO"], length(COchoices))) 
	pred, _ = KNN(kernel_matrix, train_labels)

	shots = [j for (i, j) in res.ts]
	shot_dict = Dict([a => n for (n, a) in enumerate(data_BH.tok_shots)])

	labels = [current_heating("aug", shot) for shot in shots]
	BalancedAccuracy()(pred, labels)
end

# ╔═╡ f9cb90d5-55ba-4b14-9c59-19df69ce1b49
md"### The incorrectly labelled C-LH"

# ╔═╡ 1e1dec16-7b8b-43fe-81e7-760e5c06ba3c
let
	shot = 27930
	shot_dict = Dict(res.ts .=> 1:length(res.ts))
	closeness = kernel_matrix[:, shot_dict[("aug", shot)]]
	D = DataFrame(:label => vcat(["C-LH" for _ in 1:CLn], ["C-EH" for _ in 1:CEn], ["CO" for _ in 1:COn]), :ts => vcat(CLchoices, CEchoices, COchoices), :dist => closeness)
	
	(@subset res @byrow begin
		:predict == "C-LH"
		:label == "C-EH"
	end), sort(D, :dist, rev=true)
end

# ╔═╡ bb34f990-e0f0-4a16-b521-d26531660e2e
let
	train = 34770
	test = 34841
	# 36177, 29772, 13622, 34610, 17282, 15714, 19314, 32135, 34441, 32464, 33407, 34841 
	x1_lim = 4
	x2_lim = 3.6
	
	tr_ind = findall(i -> in(i, [("aug", train)]), data_CEL.tok_shots)[1]
	te_ind = findall(i -> in(i, [("aug", test)]), data_CEL.tok_shots)[1]

	CC = round(data_CEL.cosine_cost[tr_ind, te_ind+1], digits=2)
	FTC = round(data_CEL.flat_top_cost[tr_ind, te_ind+1], digits=2)
	
	CCexp = round(exp(-CC^2 / cos_med^2), digits=2)
	FTCexp = round(exp(-FTC^2 / ft_med^2), digits=2)
	
	shot = train
	fig1 = Figure(size=(1500, 500))
	ax1 = Axis(fig1[1:2, 2:4], title="#$shot", titlesize=25)
	
	for (n, (feat, norm, axis, label)) in enumerate(zip(["IP", "PNBI"], [1e-6, 1e-7], [ax1, ax1], [L"I_p \, (10^{-6} MA)", L"P_{\mathrm{NBI}} \, (10^{-7} MA)"]))
		P = profiles(("aug", shot)..., feat)
		lines!(axis, P.t, abs.(P.y.y .* norm), linewidth=2.5, colormap=:dracula, colorrange=(1, 8), color=n, label=label)
	end
	ax1.limits = ((0.4, x1_lim), (-0.1, 2.5))
	ax1.yticks = 0:0.5:2.5
	text!(ax1, (0.5, 2.2), text = "cosine cost = $(CC)", fontsize=25)
	text!(ax1, (0.5, 1.9), text = "flat top cost = $(FTC)", fontsize=25)
	fig1[1:2, 1] = Legend(fig1, ax1, "", framevisible=false, labelsize=35)
	
	shot = test
	ax3 = Axis(fig1[1:2, 5:7], title="#$shot", titlesize=25)
	for (n, (feat, norm, axis, label)) in enumerate(zip(["IP", "PNBI"], [1e-6, 1e-7], [ax3, ax3], [L"I_p \, (10^{-6} MA)", L"P_{\mathrm{NBI}} \, (10^{-7} MA)"]))
		P = profiles(("aug", shot)..., feat)
		lines!(axis, P.t, abs.(P.y.y .* norm), linewidth=2.5, colormap=:dracula, colorrange=(1, 8), color=n, label=label)
	end
	ax3.limits = ((0.4, x2_lim), (-0.1, 2.5))
	ax3.yticks = 0:0.5:2.5
	# fig1[1:2, 8] = Legend(fig1, ax1, "", framevisible=false, labelsize=35)
	# fig1[4:5, 8] = Legend(fig1, ax2, "", framevisible=false, labelsize=35)

	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/shot_comparison.png", fig1)
	fig1
end

# ╔═╡ 6ce819bb-56d8-494f-861b-865088f0e7fa
md"# classifying baseline/ hybrid"

# ╔═╡ 9e86ed42-5b05-4c04-9d0f-011f905f4d12
function classify!(data::DTW_hyp_1, labelled_data::OrderedDict{Tuple{String, Int64}, String}, hyperparameters::Tuple; interesting::String="")

    labelled_ts = collect(labelled_data |> keys)
    labelled_y = collect(labelled_data |> values)
    shot_dict = Dict([a => n for (n, a) in enumerate(data.tok_shots)])
    labelled_ind = [shot_dict[k] for k in labelled_ts]

    labels = collect(labelled_data |> values) |> unique

    X = exp.(-(Array(data.cosine_cost[!, 2:end]) ./ (hyperparameters[1])).^2) .* 
        exp.(-(Array(data.flat_top_cost[!, 2:end]) ./ (hyperparameters[2])).^2) 

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
    return res, confidence, KK
end

# ╔═╡ f2b1fea6-5bb2-4e5f-8214-0ae5db81ec52
begin
	BH_features = ["IP", "PNBI", "PICRH", "PECRH", "BETAPOL", "Q95", "NGW", "LI"]
	BH_tss = tok_shots((@subset which(BH_features) @byrow in(:tok, ["aug"])))
end

# ╔═╡ 22dce3ab-36d5-4ec0-9596-cffbf97e2cb6
let
	shot = 32305 
	x1_lim = 8.6
	x2_lim = 4.5
	fig1 = Figure(size=(1900, 900))
	ax1 = Axis(fig1[1:3, 2:4], title="#$shot", titlesize=25)
	ax2 = Axis(fig1[4:6, 2:4])

	features = ["IP", "PNBI", "PECRH", "PICRH", "Q95", "BETAPOL", "NGW", "LI"]
	t = FTIP[("aug", shot)]
    
	for (n, (feat, norm, axis, label)) in enumerate(zip(features, [1e-6, 1e-7, 1e-7, 1e-7, 1e-1, 1, 1, 1], [ax1, ax1, ax1, ax1, ax2, ax2, ax2, ax2], [L"I_p \, (10^{-6} MA)", L"P_{\mathrm{NBI}} \, (10^{-7} MA)", L"P_{\mathrm{ECRH}} \, (10^{-7} MA)", L"P_{\mathrm{ICRH}} \, (10^{-7} MA)", L"q_{95} \, (10^{-1})", L"\beta_p", L"f_{\mathrm{GW}}", L"\ell_i"]))
		P = profiles(("aug", shot)..., feat)
		BV = t[1] .< P.t .< t[2]
		Y = abs.(P.y.y .* norm)
		x_shade = P.t[BV]
		y_shade = Y[BV]
		band!(axis, x_shade, 0, y_shade, color=(:cyan, 0.15))
		lines!(axis, P.t, abs.(P.y.y .* norm), linewidth=1.5, colormap=:dracula, colorrange=(1, 8), color=n, label=label)
	end
	ax1.limits = ((0, x1_lim), (-0.1, 2.5))
	ax2.limits = ((0, x1_lim), (-0.1, 2.5))
	ax1.yticks = 0:0.5:2.5
	ax2.yticks = 0:0.5:2.5
	fig1[1:2, 1] = Legend(fig1, ax1, "", framevisible=false, labelsize=35)
	fig1[4:5, 1] = Legend(fig1, ax2, "", framevisible=false, labelsize=35)
	
	shot = 34770
	t = FTIP[("aug", shot)]
	ax3 = Axis(fig1[1:3, 5:7], title="#$shot", titlesize=25)
	ax4 = Axis(fig1[4:6, 5:7])
	for (n, (feat, norm, axis, label)) in enumerate(zip(features, [1e-6, 1e-7, 1e-7, 1e-7, 1e-1, 1, 1, 1], [ax3, ax3, ax3, ax3, ax4, ax4, ax4, ax4], [L"I_p \, (10^{-6} MA)", L"P_{\mathrm{NBI}} \, (10^{-7} MA)", L"P_{\mathrm{ECRH}} \, (10^{-7} MA)", L"P_{\mathrm{ICRH}} \, (10^{-7} MA)", L"q_{95} \, (10^{-1})", L"\beta_p", L"f_{\mathrm{GW}}", L"\ell_i"]))
		P = profiles(("aug", shot)..., feat)
		BV = t[1] .< P.t .< t[2]
		Y = abs.(P.y.y .* norm)
		x_shade = P.t[BV]
		y_shade = Y[BV]
		band!(axis, x_shade, 0, y_shade, color=(:cyan, 0.15))
		lines!(axis, P.t, abs.(P.y.y .* norm), linewidth=1.5, colormap=:dracula, colorrange=(1, 8), color=n, label=label)
	end
	ax3.limits = ((0, x2_lim), (-0.1, 2.5))
	ax4.limits = ((0, x2_lim), (-0.1, 2.5))
	ax3.yticks = 0:0.5:2.5
	ax4.yticks = 0:0.5:2.5
	# fig1[1:2, 8] = Legend(fig1, ax1, "", framevisible=false, labelsize=35)
	# fig1[4:5, 8] = Legend(fig1, ax2, "", framevisible=false, labelsize=35)

	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/shot_comparison_stochastic_FTNBI.png", fig1)
	fig1
end

# ╔═╡ 470812a2-75e5-4cc2-a5a9-a31ee2757458
let
	signal_1 = data_BH.profile_data[("aug", 28882)]
	signal_2 = data_BH.profile_data[("aug", 7934)]
	mat = dtw_cost_matrix(signal_2, signal_1, CosineDist(), transportcost=1.1)
	cost, i1, i2 = DynamicAxisWarping.trackback(mat)

	k=3
	fig = Figure(size=(1800, 900));
	ax = Axis(fig[1:11, 3:14])
	heatmap!(mat, colormap=:thermal)
	# contour!(mat; color = :white, levels=1:7, labels=true, labelfont=:bold, labelsize = 12)
	lines!(ax, i2, i1, color=:white, linewidth=3)
	ax.xticklabelsvisible=false
	ax.yticklabelsvisible=false
	ax.xticks=1:13
	ax.xticksvisible=false
	ax.yticksvisible=false

	# text!((2, 4), text="hi")
	for (i, j) in Iterators.product(2:13:size(signal_1, 2), 1:13:size(signal_2, 2))
		text!((i-0.1, j-0.7),
		    text="$(round(mat[i, j], digits=2))",
		    color=(:white, 0.5),
			fontsize=25
		)
	end
	ax1 = Axis(fig[1:11, 1:2])

	for (i, norm, style) in zip(1:8, [1, 1E-1, 1, 1, 1, 1E-1, 1, 1], [:dash, :dash, :dash, :dash, :solid, :dash, :solid, :solid])
		trans = 1
		if style == :dash
			trans = 0.4
		end
		if maximum(signal_2[i, :]) < 0.1
			continue
		end
		lines!(ax1, -signal_2[i, :][:].*norm, 1:size(signal_2, 2), colormap=(:dracula, trans), colorrange=(1, 8), color=i, linestyle=style, label=BH_features[i])
	end
	ax1.topspinevisible=false
	ax1.leftspinevisible=false
	ax1.rightspinevisible=false
	ax1.bottomspinevisible=false
	ax1.xgridvisible=false
	ax1.ygridvisible=false
	ax1.yticklabelsvisible=false
	ax1.xticklabelsvisible=false
	ax1.yticksvisible=false
	ax1.xticksvisible=false

	ax2 = Axis(fig[12:14, 3:14])
	for (i, norm, style) in zip(1:8, [1, 1E-1, 1, 1, 1, 1E-1, 1, 1], [:dash, :dash, :dash, :dash, :solid, :dash, :solid, :solid])
		trans = 1
		if style == :dash
			trans = 0.4
		end
		if maximum(signal_2[i, :]) < 0.1
			continue
		end
		lines!(ax2, 1:size(signal_1, 2), signal_1[i, :].*norm, colormap=(:dracula, trans), colorrange=(1, 8), color=i, linestyle=style, label=BH_features[i])
	end
	ax2.topspinevisible=false
	ax2.leftspinevisible=false
	ax2.rightspinevisible=false
	ax2.bottomspinevisible=false
	ax2.xgridvisible=false
	ax2.ygridvisible=false
	ax2.yticklabelsvisible=false
	ax2.xticklabelsvisible=false
	ax2.yticksvisible=false
	ax2.xticksvisible=false

	Legend(fig[12:14, 1:2], ax2, framevisible=false, labelsize=30, nbanks=2, patchlabelgap=5, patchsize=(15, 20))

	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/BH_cosine_dist.png", fig)
	fig
end

# ╔═╡ 654889ec-9e02-4763-94f1-482a136ad042
let
	signal_1 = data_BH.profile_data[("aug", 28882)]
	signal_2 = data_BH.profile_data[("aug", 7934)]
	mat = dtw_cost_matrix(signal_2, signal_1, TotalVariation(), transportcost=1.1)
	cost, i1, i2 = DynamicAxisWarping.trackback(mat)

	k=3
	fig = Figure(size=(1800, 900));
	ax = Axis(fig[1:11, 3:14])
	heatmap!(mat, colormap=:thermal)
	# contour!(mat; color = :white, levels=1:7, labels=true, labelfont=:bold, labelsize = 12)
	lines!(ax, i2, i1, color=:white, linewidth=3)
	ax.xticklabelsvisible=false
	ax.yticklabelsvisible=false
	ax.xticks=1:13
	ax.xticksvisible=false
	ax.yticksvisible=false

	# text!((2, 4), text="hi")
	for (i, j) in Iterators.product(2:13:size(signal_1, 2), 1:13:size(signal_2, 2))
		text!((i-0.1, j-0.7),
		    text="$(round(mat[i, j], digits=2))",
		    color=(:white, 0.5)
		)
	end
	ax1 = Axis(fig[1:11, 1:2])

	for (i, norm, style) in zip(1:8, [1, 1E-1, 1, 1, 1, 1E-1, 1, 1], [:dash, :dash, :dash, :dash, :solid, :dash, :solid, :solid])
		trans = 1
		if style == :dash
			trans = 0.4
		end
		if maximum(signal_2[i, :]) < 0.1
			continue
		end
		lines!(ax1, -signal_2[i, :][:].*norm, 1:size(signal_2, 2), colormap=(:dracula, trans), colorrange=(1, 8), color=i, linestyle=style, label=BH_features[i])
	end
	ax1.topspinevisible=false
	ax1.leftspinevisible=false
	ax1.rightspinevisible=false
	ax1.bottomspinevisible=false
	ax1.xgridvisible=false
	ax1.ygridvisible=false
	ax1.yticklabelsvisible=false
	ax1.xticklabelsvisible=false
	ax1.yticksvisible=false
	ax1.xticksvisible=false

	ax2 = Axis(fig[12:14, 3:14])
	for (i, norm, style) in zip(1:8, [1, 1E-1, 1, 1, 1, 1E-1, 1, 1], [:dash, :dash, :dash, :dash, :solid, :dash, :solid, :solid])
		trans = 1
		if style == :dash
			trans = 0.4
		end
		if maximum(signal_2[i, :]) < 0.1
			continue
		end
		lines!(ax2, 1:size(signal_1, 2), signal_1[i, :].*norm, colormap=(:dracula, trans), colorrange=(1, 8), color=i, linestyle=style, label=BH_features[i])
	end
	ax2.topspinevisible=false
	ax2.leftspinevisible=false
	ax2.rightspinevisible=false
	ax2.bottomspinevisible=false
	ax2.xgridvisible=false
	ax2.ygridvisible=false
	ax2.yticklabelsvisible=false
	ax2.xticklabelsvisible=false
	ax2.yticksvisible=false
	ax2.xticksvisible=false

	Legend(fig[12:14, 1:2], ax2, framevisible=false, labelsize=15, nbanks=2, patchlabelgap=5, patchsize=(15, 20))

	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/BH_TV_dist.png", fig)
	fig
end;

# ╔═╡ 6f7b1f11-bc6b-431c-a38a-e1b166487ec5
let
	signal_1 = data_BH.flat_top_data[("aug", 28882)]
	signal_2 = data_BH.flat_top_data[("aug", 7934)]
	mat = dtw_cost_matrix(signal_2, signal_1, TotalVariation(), transportcost=1.1)
	cost, i1, i2 = DynamicAxisWarping.trackback(mat)

	k=3
	fig = Figure(size=(1800, 900));
	ax = Axis(fig[1:11, 3:14])
	heatmap!(mat, colormap=:thermal)
	# contour!(mat; color = :white, levels=1:7, labels=true, labelfont=:bold, labelsize = 12)
	lines!(ax, i2, i1, color=:white, linewidth=3)
	ax.xticklabelsvisible=false
	ax.yticklabelsvisible=false
	ax.xticks=1:13
	ax.xticksvisible=false
	ax.yticksvisible=false

	# text!((2, 4), text="hi")
	for (i, j) in Iterators.product(2:13:size(signal_1, 2), 1:13:size(signal_2, 2))
		text!((i-0.1, j-0.7),
		    text="$(round(mat[i, j], digits=2))",
		    color=(:white, 0.5)
		)
	end
	ax1 = Axis(fig[1:11, 1:2])

	for (i, norm, style) in zip(1:8, [1, 1E-1, 1, 1, 1, 1E-1, 1, 1], [:dash, :dash, :dash, :dash, :solid, :dash, :solid, :solid])
		trans = 1
		if style == :dash
			trans = 0.4
		end
		if maximum(signal_2[i, :]) < 0.1
			continue
		end
		lines!(ax1, -signal_2[i, :][:].*norm, 1:size(signal_2, 2), colormap=(:dracula, trans), colorrange=(1, 8), color=i, linestyle=style, label=BH_features[i])
	end
	ax1.topspinevisible=false
	ax1.leftspinevisible=false
	ax1.rightspinevisible=false
	ax1.bottomspinevisible=false
	ax1.xgridvisible=false
	ax1.ygridvisible=false
	ax1.yticklabelsvisible=false
	ax1.xticklabelsvisible=false
	ax1.yticksvisible=false
	ax1.xticksvisible=false

	ax2 = Axis(fig[12:14, 3:14])
	for (i, norm, style) in zip(1:8, [1, 1E-1, 1, 1, 1, 1E-1, 1, 1], [:dash, :dash, :dash, :dash, :solid, :dash, :solid, :solid])
		trans = 1
		if style == :dash
			trans = 0.4
		end
		if maximum(signal_2[i, :]) < 0.1
			continue
		end
		lines!(ax2, 1:size(signal_1, 2), signal_1[i, :].*norm, colormap=(:dracula, trans), colorrange=(1, 8), color=i, linestyle=style, label=BH_features[i])
	end
	ax2.topspinevisible=false
	ax2.leftspinevisible=false
	ax2.rightspinevisible=false
	ax2.bottomspinevisible=false
	ax2.xgridvisible=false
	ax2.ygridvisible=false
	ax2.yticklabelsvisible=false
	ax2.xticklabelsvisible=false
	ax2.yticksvisible=false
	ax2.xticksvisible=false

	Legend(fig[12:14, 1:2], ax2, framevisible=false, labelsize=15, nbanks=2, patchlabelgap=5, patchsize=(15, 20))

	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/BH_FT_TV_dist.png", fig)
	fig
end;

# ╔═╡ bb7cc056-91e0-4a3a-9c84-7abee51eb1d1
begin
	# 
    all_IB = [7643, 8127, 29619, 29620, 29621, 29622, 29623, 29636, 29953, 29956, 29957, 29958, 29962, 29963, 29964, 29965, 29976, 29977, 29979, 29980, 29981, 32103, 32104, 32105, 32106, 32109, 32110, 32111, 32112, 32113, 32114, 32115, 32120, 32121, 32122, 32123, 32463, 32464, 32471, 32472, 32474, 32475, 32489, 33370, 33371, 33374, 33375, 33376, 33377, 33406, 33407, 33409, 33427, 34441, 34442, 34443, 34444, 34445, 34446, 34449, 34450, 34454, 34489, 34490, 34840, 35975, 36177, 37081, 37094, 37096, 39124, 39125, 39126, 39128, 39129, 40411, 40851, 40410, 37093, 37082, 36178, 37080, 36176, 34842, 40206, 40207, 40202, 40203, 34838, 34839, 34841, 34844
]

	Bchoices = [("aug", shot) for shot in all_IB]
	Hchoices = [("aug", shot) for shot in [11190, 16688, 16736, 18046, 18869, 18880, 19314, 25764, 26338, 26913, 27930, 34769, 34770, 
	    34774, 34775, 34776, 36443, 36408]]
	
	Bn = length(Bchoices)
	Hn = length(Hchoices)

	DataFrame(:ts => vcat(Bchoices, Hchoices), :label => vcat(repeat(["baseline"], Bn), repeat(["hybrid"], Hn)))
end

# ╔═╡ 2fbc3d4d-adf2-4031-9bba-7299379172a2
let
	start_quantile = 0.55
	k = [21, 11]
	data = data_BH
	labelled_data = let
	    labelled_shots = vcat(Bchoices, Hchoices)
	    labels = vcat(repeat(["baseline"], length(Bchoices)),
	        repeat(["hybrid"], length(Hchoices))) 
	    d = OrderedDict([ts for ts in labelled_shots] .=> labels)
	end

	labelled_ts = collect(labelled_data |> keys)
    labelled_y = collect(labelled_data |> values)
    shot_dict = Dict([a => n for (n, a) in enumerate(data_BH.tok_shots)])
    labelled_ind = [shot_dict[k] for k in labelled_ts]

    cos_arr = Array(data.cosine_cost[:, 2:end])
    ft_arr = Array(data.flat_top_cost[:, 2:end])
	
    cos_med = quantile(cos_arr[labelled_ind, labelled_ind][:], start_quantile)
    ft_med = quantile(ft_arr[labelled_ind, labelled_ind][:], start_quantile)

	shot_dict = Dict([a => n for (n, a) in enumerate(data.tok_shots)])
	labelled_ts = collect(labelled_data |> keys)
	labelled_y = collect(labelled_data |> values)
    labels = collect(labelled_data |> values) |> unique

    cnt, ACC, δ_acc = 1, 0., 100
	N = 10
		
	figs = Figure[]
    while (cnt < 12) && (ACC !== 1.) && (δ_acc > 0.001)
		
		fs = range(ft_med-(ft_med/(cnt+0.2)), ft_med+(ft_med/(cnt+0.2)), length=3+(2*cnt))
		cs = range(cos_med-(cos_med/(cnt+0.2)), cos_med+(cos_med/(cnt+0.2)), length=3+(2*cnt))
		
		Cs = collect(Iterators.product(
			cs,
			# range(mag_med/2, mag_med*2, length=10),
			fs
		))
		
		nc, nft = size(Cs)
		hyp_search = zeros(nc, nft, 3)
		acc_int = [zeros(nc, nft) for _ in 1:nthreads()]
		
		for S in 1:N
			Random.seed!(S)
			train_ind = training_partion(labelled_data, labels, k)
			for (i, j) in Iterators.product(1:nc, 1:nft)
				CC = Cs[i, j][1]
				FTC = Cs[i, j][2]
	
				X = exp.(-(Array(data.cosine_cost[!, 2:end]) ./ CC).^2) .* 
					exp.(-(Array(data.flat_top_cost[!, 2:end]) ./ FTC).^2)
	
				K = X[labelled_ind[train_ind], labelled_ind[train_ind]]
				model = svmtrain(K, labelled_y[train_ind], kernel=Kernel.Precomputed)
	
				KK = X[labelled_ind[train_ind], labelled_ind[Not(train_ind)]]
				ỹ, _ = svmpredict(model, KK)

				acc_int[threadid()][i, j] += BalancedAccuracy()(labelled_y[Not(train_ind)], ỹ)
				hyp_search[i, j, 2] = CC
				hyp_search[i, j, 3] = FTC
			end
		end
	    hyp_search[:, :, 1] = map(+, acc_int...)
		hyp_search[:, :, 1] ./= N

		max = maximum(hyp_search[:, :, 1])
        inds = findall(i -> i == max, hyp_search[:, :, 1])
        ind = sample(inds, 1)[1]

        max = round(max, digits=5)

		(cos_med, ft_med) = (hyp_search[ind, 2], hyp_search[ind, 3])
        println(max, ": ($cos_med, $ft_med)")
        δ_acc = abs(-(ACC, max))

        ACC = max 
        # no_good_values = length(inds)
        cnt += 1

		f = Figure(size=(800, 500))
		a = Axis(f[1, 1])
		he = heatmap!(a, cs, fs, hyp_search[:, :, 1], colormap=:viridis, colorrange=(0.75, 1))
		a.xlabel=L"\sigma_1"
		a.ylabel=L"\sigma_2"
		a.xticklabelsize = 30
		a.yticklabelsize = 30
		a.xlabelsize = 40
		a.ylabelsize = 40
		
		Colorbar(f[1, 2], he, ticklabelsize=32)

		scatter!([cos_med], [ft_med], marker=:cross, color=:gray, markersize=25)
		push!(figs, f)
	end
	for (n, fig) in enumerate(figs)
		save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/grid_search_$(n).png", fig)
	end
	figs
	# hyp_search
end

# ╔═╡ 1b3bf414-6a67-46cf-b1bb-c7b4f6269594
# RES, CONFIDENCE, KERNEL, (COS_med, FT_med) = let
# 	labelled_dict = let
# 	    labelled_shots = vcat(Bchoices, Hchoices)
# 	    labels = vcat(repeat(["baseline"], length(Bchoices)),
# 	        repeat(["hybrid"], length(Hchoices))) 
# 	    d = OrderedDict([ts for ts in labelled_shots] .=> labels)
# 	end
# 	hyper_parameter_search(data_BH, labelled_dict, [64, 13], interesting="hybrid", N=250, start_quantile=0.7)
# end

# ╔═╡ ce8ed855-e3dd-4db1-b3d4-72356afb54e8
begin
	COS_med, FT_med = 20.3 ,337.7
	RES, CONFIDENCE, KERNEL = let
		labelled_dict = let
		    labelled_shots = vcat(Bchoices, Hchoices)
		    labels = vcat(repeat(["baseline"], length(Bchoices)),
		        repeat(["hybrid"], length(Hchoices))) 
		    d = OrderedDict([ts for ts in labelled_shots] .=> labels)
		end
		classify!(data_BH, labelled_dict, (COS_med, FT_med), interesting="hybrid")
	end
end

# ╔═╡ f7548393-01bb-46c7-8928-638147f81964
let
	train_tss = vcat(Bchoices, Hchoices)
	n = length(train_tss)
	ind = [data_BH.shot_dict[ts] for ts in train_tss]

	fig = Figure();
	ax = Axis(fig[1, 1], xticks=(0.7:0.2:0.7, [L"\mathrm{DTW_{cos}}|_{I_p^{80%}}"]), xticklabelsize=20)
	ax1 = Axis(fig[1, 2], xticks=(0.7:1:1.7, [L"\mathrm{DTW_{mag}}|_{I_p^{80%}}", L"\mathrm{DTW_{mag}}|_{P_{NBI}^{70%}}"]), xticklabelsize=20)

	dat = data_BH.cosine_cost
	mat = Array(dat[ind, ind.+1])
	CD = abs.(vcat([mat[i, i+1:n] for i in 1:n-1]...))
	hist!(ax, CD, scale_to=-0.6, offset=1, direction=:x, color=:gray70)
	hlines!(ax, [COS_med], xmin=0.05, xmax=0.95, color=:red)

	dat = data_BH.flat_top_cost
	mat = Array(dat[ind, ind.+1])
	CD = abs.(vcat([mat[i, i+1:n] for i in 1:n-1]...))
	hist!(ax1, CD, scale_to=-0.6, offset=2, direction=:x, color=:gray50)
	hlines!(ax1, [FT_med], xmin=0.05, xmax=0.95, color=:red)
	
	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/cost_spread_BH.png", fig)
	fig
end

# ╔═╡ 52e67055-177f-4c88-b03a-e55876c803cf
let
	pred_shots = [j for (i, j) in RES.ts]
	shot_dict = Dict([a => n for (n, a) in enumerate(data_BH.tok_shots)])

	CH = [current_heating("aug", shot) for shot in pred_shots]
	
	col_dict = Dict("baseline"=>1, "hybrid"=>2, "training"=>3)
	CO_dict = Dict("CO"=> :cross, "C-EH"=> :ltriangle, "C-LH"=> :rtriangle)

	training_D = DataFrame(:ts => vcat(Bchoices, Hchoices),
						:predict => vcat(["training" for _ in 1:length(Bchoices)],
							["training" for _ in 1:length(Hchoices)]),
						:conf => vcat([(ts[2], (0.6*cos(n+1))) for (n, ts) in enumerate(Bchoices)],
									[(ts[2], (0.4*cos(n+1))) for (n, ts) in enumerate(Hchoices)]))
	training_D.CH = [current_heating(ts...) for ts in training_D.ts]

	RES.conf = [Tuple.(zip(ts[2], CONFIDENCE[1, n])) for (n, ts) in enumerate(RES.ts)]
	
	# RES_updated = vcat(RES[!, [:ts, :predict, :conf]], training_D)
	RES.CH .= CH
	ỹ = RES.predict
	
	HYBRID = (@subset RES @byrow :predict == "hybrid")[1:4:end, :]
	CO = (@subset RES @byrow :CH == "CO")
	
	f = Figure(size=(1200, 700));
	a0 = Axis(f[1:3, 1:12])
	a = Axis(f[4:16, 1:12])
	a1 = Axis(f[17:19, 1:12])
    a_sp = scatter!(a, RES.conf; colormap=:tab10, 
		color=[col_dict[el] for el in RES.predict], 
		colorrange = (1, 3),
		strokewidth = 0.1, 
		markersize = 18,
		marker = [CO_dict[label] for label in RES.CH],
		label = [label => (;colormap=:tab10, colorrange=(1, 3), color=col_dict[label]) for label in ỹ])
	a0.xticklabelsvisible = false
	a0.yticklabelsvisible = false
	hidespines!(a0)
	a0.limits = (nothing, (-0.7, 0.7))
	hidedecorations!(a0)
	a.xticks=(10000:10000:40000, ["10,000", "20,000", "30,000", "40,000"])
	a1.yticklabelsvisible = false
	hidespines!(a1)
	a1.limits = (nothing, (-0.6, 0.6))
	hidedecorations!(a1)
	
	linkxaxes!(a0, a, a1)
	
	a0_sp = scatter!(a0, training_D.conf[1:length(Bchoices)]; colormap=:tab10, 
		color=3, 
		colorrange = (1, 3),
		strokewidth = 0.1, 
		markersize = 18,
		marker = [CO_dict[label] for label in training_D.CH[1:length(Bchoices)]],
		label = "training")

	scatter!(a1, training_D.conf[length(Bchoices)+1:end]; colormap=:tab10, 
		color=3, 
		colorrange = (1, 3),
		strokewidth = 0.1, 
		markersize = 18,
		marker = [CO_dict[label] for label in training_D.CH[length(Bchoices)+1:end]],
		label = "training")

    pos = [(:right, :bottom), (:left, :top), (:left, :bottom), (:right, :top)]
    for (n, (ts, coord)) in enumerate(zip(HYBRID.ts, HYBRID.conf))
        text!(a, coord, text="$(ts[2])", align=pos[mod(n+2,4)+1], fontsize=13)
    end
	pos = [(:right, :bottom), (:left, :top), (:left, :bottom), (:right, :top)]
    for (n, (ts, coord)) in enumerate(zip(CO.ts, CO.conf))
        text!(a, coord, text="$(ts[2])", align=pos[mod(n+1,4)+1], fontsize=13)
    end
	pos = [(:right, :bottom), (:left, :top), (:left, :bottom), (:right, :top)]
	tr_D_CO = @subset training_D @byrow :CH == "CO"
    for (n, (ts, coord)) in enumerate(zip(tr_D_CO.ts, tr_D_CO.conf))
		text!(a1, coord, text="$(ts[2])", align=pos[mod(n+2,4)+1], fontsize=13)
    end

	# extra = (@subset training_data @byrow in(:ts, [("aug", shot) for shot in [12000, 12032]]))
	# for (n, (ts, coord)) in enumerate(zip(extra.ts, extra.conf))
 #        text!(a, coord, text="$(ts[2])", align=pos[mod(n,2+2)+1], fontsize=7)
 #    end

	# a.xgridvisible = false
	# a.ygridvisible = false
	
	group_marker = [MarkerElement(marker = marker, color = :black,
	    strokecolor = :transparent,
	    markersize = 12) for marker in [:ltriangle, :rtriangle, :cross]]

	Legend(f[15, 2],
	    group_marker,
	    ["EH", "LH", "CO"],
	 	framevisible=false, labelsize=20)

	colors_3 = get(ColorSchemes.tab10, range(0.0, 1.0, length=3))
	group_marker = [MarkerElement(marker = :circ, color = colors_3[i],
	    strokecolor = :transparent,
	    markersize = 12) for i in 1:3]

    Legend(f[15, 1],
	    group_marker,
	    ["baseline", "hybrid", "training"],
	 	framevisible=false, labelsize=20)
	# axislegend(a0, unique=true, merge=true, position=:lb, labelsize=20)

	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/Baseline_Hybrid_classified.png", f)
	
	# println("hybrid = ", (@subset RES @byrow :predict == "hybrid").CH |> countmap)
	# println("baseline = ", (@subset RES @byrow :predict == "baseline").CH |> countmap)
	
    f
end

# ╔═╡ 3671c547-d8d2-455c-916a-5d7529542501
let
	f = Figure(size=(1800, 900));
	a1 = Axis(f[1:3, 1:20], title="Baseline", ylabel="testing shots", ylabelsize=20, yticks=1:50:500, ylabelpadding=20)
	a2 = Axis(f[1:3, 21:25], yticklabelsvisible=false, yticksvisible=false, title="Hybrid")
	he = heatmap!(a1, KERNEL[1:92, :])
	heatmap!(a2, KERNEL[93:end, :])
	Colorbar(f[1:3, end+1], he, ticks=0:0.1:1, width=20)
	a1.xticks = (1:92, ["#$j" for (i,j) in Bchoices])
	a2.xticks = (1:18, ["#$j" for (i,j) in Hchoices])
	
	a1.xticklabelrotation = π/2
	a2.xticklabelrotation = π/2
	
	a1.xticklabelsize = 15
	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/BH_kernel_matrix.png", f)
	f
end

# ╔═╡ 4aeeb233-308e-4763-84d3-fc8e232d1e77
let
	shot = 26881
	shot_dict = Dict(RES.ts .=> 1:length(RES.ts))
	closeness = KERNEL[:, shot_dict[("aug", shot)]]
	D = DataFrame(:label => vcat(["baseline" for _ in 1:Bn], 
								["hybrid" for _ in 1:Hn]), 
				:ts => vcat(Bchoices, Hchoices), 
				:confidence => closeness)
	sort(D, :confidence, rev=true)
end

# ╔═╡ 6f2f722c-e2cb-4d67-8bb4-f932e7f7eb8e
let
	train = 33371
	test = 26881
	# 36177, 29772, 13622, 34610, 17282, 15714, 19314, 32135, 34441, 32464, 33407, 34841 
	x1_lim = 4.15
	x2_lim = 5.9

	tr_ind = findall(i -> in(i, [("aug", train)]), data_BH.tok_shots)[1]
	te_ind = findall(i -> in(i, [("aug", test)]), data_BH.tok_shots)[1]

	CC = round(data_BH.cosine_cost[tr_ind, te_ind+1], digits=2)
	FTC = round(data_BH.flat_top_cost[tr_ind, te_ind+1], digits=2)
	
	CCexp = round(exp(-CC^2 / COS_med^2), digits=2)
	FTCexp = round(exp(-FTC^2 / FT_med^2), digits=2)
	
	shot = train
	
	fig1 = Figure(size=(1900, 900))
	ax1 = Axis(fig1[1:3, 2:4], title="#$shot", titlesize=25)
	ax2 = Axis(fig1[4:6, 2:4])
	for (n, (feat, norm, axis, label)) in enumerate(zip(["IP", "PNBI", "PECRH", "PICRH", "Q95", "BETAPOL", "NGW", "LI"], [1e-6, 1e-7, 1e-7, 1e-7, 1e-1, 1, 1, 1], [ax1, ax1, ax1, ax1, ax2, ax2, ax2, ax2], [L"I_p \, (10^{-6} MA)", L"P_{\mathrm{NBI}} \, (10^{-7} MA)", L"P_{\mathrm{ECRH}} \, (10^{-7} MA)", L"P_{\mathrm{ICRH}} \, (10^{-7} MA)", L"q_{95} \, (10^{-1})", L"\beta_p", L"f_{\mathrm{GW}}", L"\ell_i"]))
		P = profiles(("aug", shot)..., feat)
		lines!(axis, P.t, abs.(P.y.y .* norm), linewidth=2.5, colormap=:dracula, colorrange=(1, 8), color=n, label=label)
	end
	ax1.limits = ((0., x1_lim), (-0.1, 2.5))
	ax2.limits = ((0., x1_lim), (-0.1, 2.5))
	ax1.yticks = 0:0.5:2.5
	ax2.yticks = 0:0.5:2.5
	text!(ax1, (0.1, 2.2), text = "cosine cost = $(CCexp)", fontsize=25)
	text!(ax1, (0.1, 1.9), text = "flat top cost = $(FTCexp)", fontsize=25)
	fig1[1:2, 1] = Legend(fig1, ax1, "", framevisible=false, labelsize=35)
	fig1[4:5, 1] = Legend(fig1, ax2, "", framevisible=false, labelsize=35)
	
	shot = test
	ax3 = Axis(fig1[1:3, 5:7], title="#$shot", titlesize=25)
	ax4 = Axis(fig1[4:6, 5:7])
	for (n, (feat, norm, axis, label)) in enumerate(zip(["IP", "PNBI", "PECRH", "PICRH", "Q95", "BETAPOL", "NGW", "LI"], [1e-6, 1e-7, 1e-7, 1e-7, 1e-1, 1, 1, 1], [ax3, ax3, ax3, ax3, ax4, ax4, ax4, ax4], [L"I_p \, (10^{-6} MA)", L"P_{\mathrm{NBI}} \, (10^{-7} MA)", L"P_{\mathrm{ECRH}} \, (10^{-7} MA)", L"P_{\mathrm{ICRH}} \, (10^{-7} MA)", L"q_{95} \, (10^{-1})", L"\beta_p", L"f_{\mathrm{GW}}", L"\ell_i"]))
		P = profiles(("aug", shot)..., feat)
		lines!(axis, P.t, abs.(P.y.y .* norm), linewidth=2.5, colormap=:dracula, colorrange=(1, 8), color=n, label=label)
	end
	ax3.limits = ((0., x2_lim), (-0.1, 2.5))
	ax4.limits = ((0., x2_lim), (-0.1, 2.5))
	ax3.yticks = 0:0.5:2.5
	ax4.yticks = 0:0.5:2.5
	# fig1[1:2, 8] = Legend(fig1, ax1, "", framevisible=false, labelsize=35)
	# fig1[4:5, 8] = Legend(fig1, ax2, "", framevisible=false, labelsize=35)

	# save("/Users/joe/Project/Papers_clean/Hybrid_classification_TH_T_24_02_22/figures/shot_comparison.png", fig1)
	fig1
end

# ╔═╡ 8d76be2c-3ee8-4302-8b83-b97e6be999e9
md"# Profiles

- toroidal magnetic field
The general form of the toroidal magnetic field simply follows from Amperes law:

$2\pi RB_\phi = \mu_0 I_T \longrightarrow B_\phi \propto \frac{I_T}{R}$
"

# ╔═╡ 94c20917-576c-491e-9cce-7e612461b5f1
tor_field(Ip::Float64, a::Float64, R::Float64)

# ╔═╡ Cell order:
# ╟─7de65134-7137-412d-96dc-9c84bfb5fb3c
# ╟─488ce488-03e6-11f0-1466-c9a64e167aec
# ╟─7c89f02a-7441-4cba-913d-35ce7d45ace9
# ╟─dc5be29b-9eb3-4877-bb61-d9766172f585
# ╟─2cd36f54-c0c6-4f8b-821a-19f2614a8475
# ╟─bed62eae-90d7-4660-9e69-6121c866b0fc
# ╟─6ced956a-38ac-4a86-9b23-1c469e8b3a95
# ╟─6a5bf655-ae06-4ef4-b44b-d0d55d729198
# ╟─e00cc312-3368-4826-a48a-ecdd698d88ce
# ╟─82ac976d-ef0e-4736-96be-fa1aad8ddaec
# ╟─1cc123b1-7c38-4083-b4b4-b3577cf0644c
# ╟─54a0e9ca-2c72-4262-bd2c-8722b67aa8fa
# ╟─b4e87f16-7061-4d41-8cde-d592d7899f66
# ╟─fd21db59-a85e-4280-919c-2c77481d7c3e
# ╟─a26ff402-ec87-4286-bdcd-d0159538d8db
# ╟─891e88be-17fd-4ab4-bcbd-e61312f56c93
# ╟─1e205f99-ff30-4c90-b0eb-76e528819cab
# ╟─eb33a1b9-d8a0-4459-a1f8-7f3a22654d63
# ╟─2fbc3d4d-adf2-4031-9bba-7299379172a2
# ╟─c77f7e68-d4b7-4141-8260-f0b4eb14df19
# ╟─bf9d1595-2a00-4232-98ac-d4c73a850568
# ╟─208cf523-8ac5-403e-a835-5f6efac2544e
# ╟─c6b42a10-22ae-4a84-b265-fc3e0aa4b4fc
# ╟─aed01189-8018-43da-b275-0b39b84c396d
# ╟─a70a39dc-9dbf-4739-aa92-f28d7aad6cc3
# ╟─f162287a-6759-4fed-9767-1a86ab427f7b
# ╠═4f098996-b1d6-4c16-8068-0da5c2f7bdd7
# ╟─4b08c5c7-f319-4544-91a0-00a2597dd64f
# ╟─e5763965-6c47-4e4b-97da-de0d63455a9c
# ╟─f9cb90d5-55ba-4b14-9c59-19df69ce1b49
# ╟─1e1dec16-7b8b-43fe-81e7-760e5c06ba3c
# ╟─bb34f990-e0f0-4a16-b521-d26531660e2e
# ╟─6ce819bb-56d8-494f-861b-865088f0e7fa
# ╟─9e86ed42-5b05-4c04-9d0f-011f905f4d12
# ╟─f2b1fea6-5bb2-4e5f-8214-0ae5db81ec52
# ╠═22dce3ab-36d5-4ec0-9596-cffbf97e2cb6
# ╟─470812a2-75e5-4cc2-a5a9-a31ee2757458
# ╟─654889ec-9e02-4763-94f1-482a136ad042
# ╟─6f7b1f11-bc6b-431c-a38a-e1b166487ec5
# ╟─bb7cc056-91e0-4a3a-9c84-7abee51eb1d1
# ╟─1b3bf414-6a67-46cf-b1bb-c7b4f6269594
# ╟─ce8ed855-e3dd-4db1-b3d4-72356afb54e8
# ╟─f7548393-01bb-46c7-8928-638147f81964
# ╟─b33afb7b-a9a4-4f60-8719-1e1843825801
# ╟─52e67055-177f-4c88-b03a-e55876c803cf
# ╠═3671c547-d8d2-455c-916a-5d7529542501
# ╠═4aeeb233-308e-4763-84d3-fc8e232d1e77
# ╟─6f2f722c-e2cb-4d67-8bb4-f932e7f7eb8e
# ╟─8d76be2c-3ee8-4302-8b83-b97e6be999e9
# ╟─94c20917-576c-491e-9cce-7e612461b5f1
