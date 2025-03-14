#!/usr/bin/env bash

julia --project=.. Bootstrap.jl time_grids_inf_100.jld2 -k 100 -g time &
julia --project=.. Bootstrap.jl freq_even_grids_inf_100.jld2 -k 100 -g freq_even &
julia --project=.. Bootstrap.jl freq_odd_grids_inf_100.jld2 -k 100 -g freq_odd

# cp time_grids_inf_100.jld2 ../data/
# cp freq_even_grids_inf_100.jld2 ../data/
# cp freq_odd_grids_inf_100.jld2 ../data/
