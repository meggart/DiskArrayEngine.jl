abstract type ProcFilter end
struct AllMissing <: ProcFilter end
struct NValid <: ProcFilter
    n::Int
end
struct AnyMissing <: ProcFilter end
struct AnyOcean <: ProcFilter end
struct NoFilter <: ProcFilter end
struct StdZero <: ProcFilter end
struct UserFilter{F} <: ProcFilter
    f::F
end

checkskip(::NoFilter, x) = false
checkskip(::AllMissing, x) = all(ismissing, x)
checkskip(::AnyMissing, x) = any(ismissing, x)
checkskip(nv::NValid, x) = count(!ismissing, x) <= nv.n
checkskip(uf::UserFilter, x) = uf.f(x)
checkskip(::StdZero, x) = all(i -> i == x[1], x)
docheck(pf::ProcFilter, x)::Bool = checkskip(pf, x)
docheck(pf::Tuple, x) = reduce(|, map(i -> docheck(i, x), pf))

getprocfilter(f::Function) = (UserFilter(f),)
getprocfilter(pf::ProcFilter) = (pf,)
getprocfilter(pf::NTuple{N,<:ProcFilter}) where {N} = pf