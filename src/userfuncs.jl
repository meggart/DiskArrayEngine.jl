
export create_userfunction
struct UserOp{F,R,I,FILT,FIN,B,T,A,KW}
    f::F
    red::R
    init::I
    filters::FILT
    finalize::FIN
    buftype::B
    outtype::T
    args::A
    kwargs::KW
end


struct Mutating end
struct NonMutating end

struct ElementFunction{F,M}
    f::F
    m::M
end
struct BlockFunction{F,M,D}
    f::F
    m::M
    dims::Val{D}
end
BlockFunction(f;mutating=false,dims) = BlockFunction(f,mutating ? Mutating() : NonMutating(),Val(dims))
getdims(::BlockFunction{<:Any,<:Any,D}) where D = D

tupelize(x,outtypes,_) = ntuple(_->x,length(outtypes))
function tupelize(x::Tuple,outtypes,s) 
    length(x)==length(outtypes) || throw(ArgumentError("Length of $s does not equal number of outputs $(length(outtypes))"))
    x
end

function create_userfunction(
        f,
        outtypes;
        is_blockfunction = false,
        is_mutating = false,
        red = nothing,
        init = zero.(outtypes),
        filters = NoFilter(),
        finalize = identity,
        buftype = outtypes,
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        dims = (),
    )
    isa(outtypes,Tuple) || (outtypes = (outtypes,))
    init = tupelize(init,outtypes,"init")
    finalize = tupelize(finalize,outtypes,"finalize")
    buftype = tupelize(buftype,outtypes,"buftype")
    isa(dims,Int) && (dims = (dims,))
    !isa(filters,Tuple) && (filters = (filters,))
    m = is_mutating ? Mutating() : NonMutating()
    uf = is_blockfunction ? BlockFunction(f,m,Val(dims)) : ElementFunction(f,m)
    UserOp(uf,red,init,filters,finalize,buftype,outtypes,args,kwargs)
end 





applyfilter(f::UserOp,myinwork) = broadcast(docheck, f.filters, myinwork)
apply_function(f::UserOp{<:ElementFunction{<:Any,<:Mutating}},xout,xin) = f.f.f(xout...,xin...,f.args...;f.kwargs...)
function apply_function(f::UserOp{<:ElementFunction{<:Any,<:NonMutating},Nothing},xout,xin)
    r = f.f.f(xin...,f.args...;f.kwargs...)
    if length(xout) == 1
        first(xout) .= r
    else
        foreach(xout,r) do x,y
            x.=y
        end
    end
end
function apply_function(f::UserOp{<:ElementFunction{<:Any,<:NonMutating},<:Base.Callable},xout,xin)
    r = f.f.f(xin...,f.args...;f.kwargs...)
    if length(xout) == 1
        first(xout)[] = f.red(first(xout)[],r)
    else
        rr = f.red(xout,r)
        foreach(xout,rr) do x,y
            x.=y
        end
    end
end