struct ArrayBuffer{A,O,LW}
    a::A
    offsets::O
    lw::LW
end

getdata(c::ArrayBuffer) = c.a
getloopinds(b::ArrayBuffer) = getloopinds(b.lw)

"Determine the needed buffer size for a given Input array and loop ranges lr"
function getbufsize(ia, lr)
    map(mysub(ia,lr.members),ia.lw.windows.members) do cr,op
        maximum(c->internal_size(op[c]),cr)
    end
end

"Creates buffers for input arrays"
function generate_inbuffers(inars,loopranges)
    map(inars) do ia
        et = eltype(ia.a)
        #@show loopranges
        Array{et}(undef,getbufsize(ia,loopranges))
    end
end

array_from_init(::Nothing,buftype,bufsize) = zeros(buftype,bufsize)
array_from_init(init::Base.Function,buftype,bufsize) = buftype[init() for _ in CartesianIndices(bufsize)]
array_from_init(init,buftype,bufsize) = buftype[init for _ in CartesianIndices(bufsize)]

#buftype_from_init(_,ia) =

"Create buffer for single output"
function generate_raw_outbuffer(init,buftype,bufsize) 
    @show buftype
    array_from_init(init,buftype,bufsize)
end

bufferrepeat(ia,loopranges) = prod(size(loopranges)) ÷ prod(mysub(ia,size(loopranges)))


"Creates buffers for all outputs"
function generate_outbuffers(outars,func,loopranges)
    generate_outbuffer_collection.(outars,func.init,func.buftype,(loopranges,))
end

offset_from_range(r) = first(r) .- 1

"Reads data from input array `ia` along domain `r` into `buffer`"
function read_range(r,ia,buffer)
    fill!(buffer,zero(eltype(buffer)))
    inds = get_bufferindices(r,ia)
    buffer[Base.OneTo.(length.(inds))...] = ia.a[inds...]
    ArrayBuffer(buffer,offset_from_range.(inds),ia.lw)
end

function get_bufferindices(r,outspecs)
    mywindowrange = mysub(outspecs,r)
    map(outspecs.lw.windows.members,mywindowrange) do w,r
        i = w[r]
        first(first(i)):last(last(i))
    end
end

struct OutputAggregator{K,V,N}
    buffers::Dict{K,V}
    bufsize::NTuple{N,Int}
    nrep::Int
end
  
function generate_outbuffer_collection(ia,init,buftype,loopranges) 
    nd = getsubndims(ia)
    bufsize = getbufsize(ia,loopranges)
    nrep = bufferrepeat(ia,loopranges)
    d = Dict{NTuple{nd,Int},Tuple{Base.RefValue{Int},Array{buftype,nd}}}()
    OutputAggregator(d,bufsize,nrep)
end

"Wraps output buffer into an ArrayBuffer"
function wrap_outbuffer(r,ia,outspecs,init,buftype,buffer::OutputAggregator)
    inds = get_bufferindices(r,outspecs)
    offsets = offset_from_range.(inds)
    n,b = get!(buffer.buffers,offsets) do 
        buf = generate_raw_outbuffer(init,buftype,buffer.bufsize)
        (Ref(0),buf)
    end
    n[] = n[]+1 
    ArrayBuffer(b,offsets,outspecs.lw)
end

"Check if maximum number of aggregations has happened for a buffer"
mustwrite(buf, bufdict) = first(bufdict.buffers[buf.offsets])[] == bufdict.nrep

"Checks if output buffers have accumulated to the end and exports to output array"
function put_buffer(r, fin, bufnow, bufferdict, ia, piddir)
  if mustwrite(bufnow,bufferdict)
    inds = get_bufferindices(r,bufnow)
    offsets = offset_from_range.(inds)
    i1 = first.(axes(ia))
    i2 = last.(axes(ia))
    skip1 = max.(0,i1.-first.(inds))
    skip2 = max.(0,last.(inds).-i2)
    inds2 = range.(first.(inds).+skip1,last.(inds).-skip2)
    r2 = range.(1 .+ skip1, skip1 .+ length.(inds2))
    if piddir !== nothing
        @debug "$(myid()) acquiring lock $piddir to write to $inds2"
        mkpidlock(piddir,wait=true,stale_age=100) do
            broadcast!(fin,view(ia,inds2...),bufnow.a[r2...])
        end
    else
        broadcast!(fin,view(ia,inds2...),bufnow.a[r2...])
    end
    delete!(bufferdict.buffers,offsets)
    true
  else
    false
  end
end

function create_buffers(inars, outars, f, loopranges)
    inbuffers_pure = generate_inbuffers(inars,loopranges)
    outbuffers_pure = generate_outbuffers(outars,f,loopranges)
    inbuffers_pure, outbuffers_pure
end