
export create_userfunction
struct UserOp{F,R,I,FILT,FIN,B,T}
    f::F
    red::R
    init::I
    filters::FILT
    finalize::FIN
    buftype::B
    outtype::T
    allow_threads::Bool
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

struct CapturedArgsFunc{F,A,KW}
    f::F
    args::A
    kwargs::KW
end
(c::CapturedArgsFunc)(x) = c.f(x,c.args...;c.kwargs...)
(c::CapturedArgsFunc)(x1,x2) = c.f(x1,x2,c.args...;c.kwargs...)
(c::CapturedArgsFunc)(x1,x2,x3) = c.f(x1,x2,x3,c.args...;c.kwargs...)
(c::CapturedArgsFunc)(x1,x2,x3,x4) = c.f(x1,x2,x3,x4,c.args...;c.kwargs...)
(c::CapturedArgsFunc)(x1,x2,x3,x4,x5) = c.f(x1,x2,x3,x4,x5,c.args...;c.kwargs...)
(c::CapturedArgsFunc)(x1,x2,x3,x4,x5,x6) = c.f(x1,x2,x3,x4,x5,x6,c.args...;c.kwargs...)
(c::CapturedArgsFunc)(x...) = c.f(x...,c.args...;c.kwargs...)
Base.show(io::IO,f::CapturedArgsFunc) = show(io,f.f)

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
        allow_threads=true,
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
    if !isempty(args) || !isempty(kwargs)
        f = CapturedArgsFunc(f,args,kwargs)
    end
    uf = is_blockfunction ? BlockFunction(f,m,Val(dims)) : ElementFunction(f,m)
    UserOp(uf,red,init,filters,finalize,buftype,outtypes,allow_threads)
end 





applyfilter(f::UserOp,myinwork) = broadcast(docheck, f.filters, myinwork)
function apply_function(f::UserOp{<:ElementFunction{<:Any,<:Mutating}},xout,xin) 
    f.f.f(xout...,xin...)
end
function apply_function(f::UserOp{<:ElementFunction{<:Any,<:NonMutating},Nothing},xout,xin)
    r = f.f.f(xin...)
    if length(xout) == 1
        first(xout) .= r
    else
        foreach(xout,r) do x,y
            x.=y
        end
    end
end
function apply_function(f::UserOp{<:ElementFunction{<:Any,<:NonMutating},<:Base.Callable},xout,xin)
    r = f.f.f(xin...)
    if length(xout) == 1
        first(xout)[] = f.red(first(xout)[],r)
    else
        rr = f.red(xout,r)
        foreach(xout,rr) do x,y
            x.=y
        end
    end
end