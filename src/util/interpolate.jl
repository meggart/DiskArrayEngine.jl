using Interpolations: BSpline, Linear, NoInterp, extrapolate, interpolate, Flat
export interpolate_diskarray, interpolate_diskarray!

function getinterpinds(oldvals::AbstractRange, newvals::AbstractRange)
    (newvals.-first(oldvals))./step(oldvals).+1
end
function getinterpinds(r1,r2)
    rev = issorted(r1) ? false : issorted(r1,rev=true) ? true : error("Axis values are not sorted")
    map(r2) do ir
        ii = searchsortedfirst(r1,ir,rev=rev)
        ii1 = max(min(ii-1,length(r1)),1)
        ii2 = max(min(ii,length(r1)),1)
        ind = if ii1 == ii2
            Float64(ii1)
        else
            ii1+(ir-r1[ii1])/(r1[ii2]-r1[ii1])
        end
        ind
    end
end


function getallsteps(xcoarse,xfine)
    interpinds = getinterpinds(xcoarse, xfine)
    resout = UnitRange{Int}[]
    icur = 1
    while icur <= length(interpinds)
        i1 = floor(interpinds[icur])
        inext = searchsortedlast(interpinds,i1+1)
        push!(resout,icur:inext)
        icur = inext+1
    end
    
    resin = DiskArrayEngine.MovingWindow(floor(Int,first(interpinds)),1,2,length(resout))
    interpinds, resin, resout
end

function interpolate_block!(xout, data, nodes...; dims,method=Linear(),threaded=false)
    innodes = axes(data)
    isinterp = ntuple(in(dims),ndims(data))
    modes = map(isinterp) do m
        m ? BSpline(method) : NoInterp()
    end
    outnodes = innodes
    for d in dims
        n,nodes = first(nodes),Base.tail(nodes)
        outnodes = Base.setindex(outnodes,n,d)
    end
    itp = extrapolate(interpolate(data,modes), Flat())
    fill_nodes!(itp,xout,outnodes)
end
@noinline function fill_nodes!(itp,xout,outnodes)
    pa = ProductArray(outnodes)
    broadcast!(Base.splat(itp),xout,pa)
end

"""
    interpolate_diskarray(a, conv)

Function to interpolate a diskarray along one or more dimensions. The interpolation is specifed by the list
`conv` consisting of pairs of `dim_index => (source_axis,dest_axis)` values. For example, to interpolate  an 
input array with dimensions (x,y,t) to new coordinates (x2,y2,t2) you can do

````julia
#Old coordinates
a = [i+j+k for i in 1:4, j in 1:5, k in 1:6]
#source coordinates
x = 5.0:5.0:20.0
y = 2.0:3.0:14.0
#target coordinates
x2 = 5.0:0.5:20.0
y2 = 1.5:1.0:14.5
r = interpolate_diskarray(a,(1=>(x,x2),2=>(y,y2)))
"""
function interpolate_diskarray(a,conv;method=Linear(),outspecs=nothing)
    allinfo = [k=>getallsteps(v...) for (k,v) in conv]
    inwindows = Base.OneTo.(size(a))
    outwindows = Base.OneTo.(size(a))
    outsize = size(a)
    dims = ()
    addarrays = ()
    sort!(allinfo,by=first)
    for (i,b) in allinfo
        inwindows = Base.setindex(inwindows,b[2],i)
        dims = (dims...,i)
        addarrays = (addarrays...,InputArray(b[1],windows=(b[3],),dimsmap=(i,)))
        outwindows = Base.setindex(outwindows,b[3],i)
        outsize = Base.setindex(outsize,length(b[1]),i)
    end
    outars = if outspecs === nothing
        (create_outwindows(outsize,windows=outwindows),)
    else
        (create_outwindows(outsize;windows=outwindows,outspecs...),)
    end
    inars = (InputArray(a,windows=inwindows), addarrays...)
    f = create_userfunction(interpolate_block!,Union{Float32,Missing},is_blockfunction=true,is_mutating=true,dims=dims,kwargs=(;method=method))
    
    optotal = GMDWop(inars, outars, f)
    r, = results_as_diskarrays(optotal)
    r
end
function interpolate_diskarray!(aout, a, conv)
    c = eachchunk(aout)
    outspecs = (ismem=false, chunks=DiskArrays.approx_chunksize(c))
    r = interpolate_diskarray(a,conv;outspecs)
    lr = DiskArrayEngine.optimize_loopranges(r.op,5e8,tol_low=0.2,tol_high=0.05,max_order=2)
    r = DiskArrayEngine.LocalRunner(r.op,lr,(aout,),threaded=true)
    run(r)
end




