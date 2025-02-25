mutable struct MinimaxGrid{T}
  n::Int64
  coefs::Vector{T}
  exponents::Vector{T}
  interp_pts::Vector{T}
  extrema::Vector{T}
  err::T
  R::T
end

convert(::Type{MinimaxGrid{K}}, grd::MinimaxGrid{T}) where {K,T} = MinimaxGrid{K}(grd.n, K.(grd.coefs), K.(grd.exponents), K.(grd.interp_pts), K.(grd.extrema), K(grd.err), K(grd.R))

convert(::Type{Dict}, grd::MinimaxGrid{T}) where {T} = Dict(
  "n" => grd.n,
  "coefs" => grd.coefs,
  "exponents" => grd.exponents,
  "interp_pts" => grd.interp_pts,
  "extrema" => grd.extrema,
  "err" => grd.err,
  "R" => grd.R
)

convert(::Type{MinimaxGrid{T}}, dict::Dict) where {T} = MinimaxGrid{T}(
  dict["n"],
  dict["coefs"],
  dict["exponents"],
  dict["interp_pts"],
  dict["extrema"],
  dict["err"],
  dict["R"]
)

function write_grid(f_or_grp, grd_key, grd)
  grd_grp = JLD2.Group(f_or_grp, grd_key)
  grd_grp["n"] = grd.n
  grd_grp["coefs"] = grd.coefs
  grd_grp["exponents"] = grd.exponents
  grd_grp["interp_pts"] = grd.interp_pts
  grd_grp["extrema"] = grd.extrema
  grd_grp["err"] = grd.err
  grd_grp["R"] = grd.R
end