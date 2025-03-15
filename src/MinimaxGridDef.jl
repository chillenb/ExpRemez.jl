module MinimaxGridDef
using JLD2
import Base: convert

export MinimaxGrid
export write_grid

mutable struct MinimaxGrid{T}
  n::Int64
  coefs::Vector{T}
  gridpts::Vector{T}
  interp_pts::Vector{T}
  extrema::Vector{T}
  err::T
  R::T
end

convert(::Type{MinimaxGrid{K}}, grd::MinimaxGrid{T}) where {K,T} = MinimaxGrid{K}(grd.n, K.(grd.coefs), K.(grd.gridpts), K.(grd.interp_pts), K.(grd.extrema), K(grd.err), K(grd.R))

convert(::Type{Dict}, grd::MinimaxGrid{T}) where {T} = Dict(
  "n" => grd.n,
  "coefs" => grd.coefs,
  "gridpts" => grd.gridpts,
  "interp_pts" => grd.interp_pts,
  "extrema" => grd.extrema,
  "err" => grd.err,
  "R" => grd.R
)

convert(::Type{MinimaxGrid{T}}, dict::Dict) where {T} = MinimaxGrid{T}(
  dict["n"],
  dict["coefs"],
  dict["gridpts"],
  dict["interp_pts"],
  dict["extrema"],
  dict["err"],
  dict["R"]
)

function write_grid(f_or_grp, grd_key, grd)
  grd_grp = JLD2.Group(f_or_grp, grd_key)
  grd_grp["n"] = grd.n
  grd_grp["coefs"] = grd.coefs
  grd_grp["gridpts"] = grd.gridpts
  grd_grp["interp_pts"] = grd.interp_pts
  grd_grp["extrema"] = grd.extrema
  grd_grp["err"] = grd.err
  grd_grp["R"] = grd.R
end

end # module MinimaxGrid
