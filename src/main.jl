using Pkg
Pkg.activate("/Users/joe/Project/Coding_clean/J_Hybrid_plasma_classification_25_02_24/src/")

using 
    # Integrals,
    CSV,
    DataFrames,
    OrderedCollections,
    DataFramesMeta,
    LinearAlgebra,
    MLJ,
    # CairoMakie,
    # GLMakie,
    GLM,
    # AlgebraOfGraphics,
    Unicode,
    ColorSchemes,
    # MLJXGBoostInterface, 
    # MLJBase,
    # Missings, 
    # Term,
    Random,
    LaTeXStrings,
    # Imbalance,
    Combinatorics,
    DynamicAxisWarping, 
    Distances, 
    StatsBase, 
    # SignalAlignment,
    # PairPlots,
    ProgressBars,
    LIBSVM,
    ThreadTools

import DataFrames: select

include("pre_defined_variables.jl")
include("extending_functions.jl")
include("preprocessing.jl")
include("H_mode_confinement.jl")
include("structures.jl")
include("structure_functions.jl")
include("scaling.jl")
# include("stationary_2D.jl")
include("secondry.jl")
include("plotting.jl")

global_database = CSV.read("/Users/joe/.data/Multi_Machine_Fusion_Data_ITER_21_03_01/DB52P3_ed_!S_good_ids.csv", DataFrame, stringtype=String)
SELDB5 = H_mode_data("/Users/joe/.data/Multi_Machine_Fusion_Data_ITER_21_03_01/DB52P3_ed_!S_good_ids.csv", :SELDB5)

DATA_ = hybrid_classification(SELDB5.original_space.ELMy)
id!(DATA_)

FTIP = flat_top_IP(tok_shots(which(["IP"])), 0.8)
FTNBI = flat_top_NBI(tok_shots(which(["PNBI"])), 0.7)

data_CEL = let
    CEL_features = ["PNBI"]
    CEL_tss = tok_shots((@subset which(CEL_features) @byrow in(:tok, ["aug"])))
    DTW_hyp_1(CEL_tss, CEL_features, 10, L=100)
end
data_BH = let
    BH_features = ["IP", "PNBI", "PICRH", "PECRH", "BETAPOL", "Q95", "NGW"]
    BH_tss = tok_shots((@subset which(BH_features) @byrow in(:tok, ["aug"])))
    DTW_hyp_1(BH_tss, BH_features, 10; L=100, ft_ind=[5,6,7])
end

classification_v1!()
# classification_v2!()