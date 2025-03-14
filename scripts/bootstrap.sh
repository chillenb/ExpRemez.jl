#!/usr/bin/env bash

julia --project=.. Bootstrap.jl time_grids_inf_100.jld2 100 time &
julia --project=.. Bootstrap.jl freq_even_grids_inf_100.jld2 100 freq_even &
julia --project=.. Bootstrap.jl freq_odd_grids_inf_100.jld2 100 freq_odd

# cp time_grids_inf_100.jld2 ../data/
# cp freq_even_grids_inf_100.jld2 ../data/
# cp freq_odd_grids_inf_100.jld2 ../data/
