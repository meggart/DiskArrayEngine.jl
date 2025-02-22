module KeyConvertDicts
using OnlineStats: OnlineStat, CountMap, OnlineStats
struct KeyConvertDict{K,V,F,INV,L} <: AbstractDict{K,V}
    f::F
    inv::INV
    values::Vector{V}
end
plusone(x) = x + 1
minusone(x) = x - 1
Base.length(kv::KeyConvertDict) = length(kv.values)
Base.getindex(kv::KeyConvertDict, i) = kv.values[kv.f(i)]
Base.setindex!(kv::KeyConvertDict, v, i) = setindex!(kv.values, v, kv.f(i))
function Base.get!(::Union{Function,Type}, kv::KeyConvertDict, i)
    return getindex(kv, i)
end
function Base.get(kv::KeyConvertDict, i, _)
    return getindex(kv, i)
end
function Base.iterate(kv::KeyConvertDict, s=1)
    if s > length(kv.values)
        nothing
    else
        k = kv.inv(s)
        (k => kv.values[s], s + 1)
    end
end
OnlineStats.CountMap{T,DT}() where {T,DT} = CountMap{T,DT}(DT(), 0)
function KeyConvertDict{K,V,F,INV,L}() where {K,V,F,INV,L}
    return KeyConvertDict{K,V,F,INV,L}(F.instance, INV.instance, zeros(V, L))
end
KeyDictType(T1, T2, f, finv, L) = KeyConvertDict{T1,T2,typeof(f),typeof(finv),L}
const NonHashCountMap = OnlineStats.CountMap{
    UInt8,KeyDictType(UInt8, Int, plusone, minusone, 256)
}
end
