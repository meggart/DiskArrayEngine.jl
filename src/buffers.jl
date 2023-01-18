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
        Array{et}(undef,getbufsize(ia,loopranges))
    end
end

array_from_init(::Nothing,ia,bufsize) = zeros(eltype(ia.a),bufsize)
array_from_init(init::Function,_,bufsize) = map(_->init(),CartesianIndices(bufsize))
array_from_init(init,_,bufsize) = fill(init,bufsize)

buftype_from_init(::Nothing,ia) = eltype(ia.a)
buftype_from_init(init::Function,_) = typeof(init())
buftype_from_init(init,_) = typeof(init)

"Create buffer for single output"
function generate_raw_outbuffer(ia,func,bufsize) 
    array_from_init(func.init,ia,bufsize)
end

bufferrepeat(ia,loopranges) = prod(size(loopranges)) ÷ prod(mysub(ia,size(loopranges)))


"Creates buffers for all outputs"
function generate_outbuffers(outars,func,loopranges)
    generate_outbuffer_collection.(outars,(func,),(loopranges,))
end

offset_from_range(r) = first(r) .- 1

"Reads data from input array `ia` along domain `r` into `buffer`"
function read_range(r,ia,buffer)
    fill!(buffer,zero(eltype(buffer)))
    inds = get_bufferindices(r,ia)
    buffer[Base.OneTo.(length.(inds))...] = ia.a[inds...]
    ArrayBuffer(buffer,offset_from_range.(inds),ia.lw)
end

function get_bufferindices(r,ia)
    mywindowrange = mysub(ia,r)
    map(ia.lw.windows.members,mywindowrange) do w,r
        i = w[r]
        first(first(i)):last(last(i))
    end
end

struct OutputAggregator{K,V,N}
    buffers::Dict{K,V}
    bufsize::NTuple{N,Int}
    nrep::Int
end
  
function generate_outbuffer_collection(ia,func,loopranges) 
    nd = getsubndims(ia)
    et = buftype_from_init(func.init,ia)
    bufsize = getbufsize(ia,loopranges)
    nrep = bufferrepeat(ia,loopranges)
    d = Dict{NTuple{nd,Int},Tuple{Base.RefValue{Int},Array{et,nd}}}()
    OutputAggregator(d,bufsize,nrep)
end

"Wraps output buffer into an ArrayBuffer"
function wrap_outbuffer(r,ia,f,buffer::OutputAggregator)
    inds = get_bufferindices(r,ia)
    offsets = offset_from_range.(inds)
    n,b = get!(buffer.buffers,offsets) do 
        buf = generate_raw_outbuffer(ia,f,buffer.bufsize)
        (Ref(0),buf)
    end
    n[] = n[]+1 
    ArrayBuffer(b,offsets,ia.lw)
end

"Check if maximum number of aggregations has happened for a buffer"
mustwrite(buf, bufdict) = first(bufdict.buffers[buf.offsets])[] == bufdict.nrep

"Checks if output buffers have accumulated to the end and exports to output array"
function put_buffer(r, f, bufnow, bufferdict, ia)
  if mustwrite(bufnow,bufferdict)
    inds = get_bufferindices(r,bufnow)
    offsets = offset_from_range.(inds)
    broadcast!(f.finalize,view(ia.a,inds...),bufnow.a[Base.OneTo.(length.(inds))...])
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