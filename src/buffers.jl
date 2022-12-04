struct ArrayBuffer{A,W,O,IL}
    a::A
    windows::W
    offsets::O
    lr::Val{IL}
end

getdata(c::ArrayBuffer) = c.a
getloopinds(::ArrayBuffer{<:Any,<:Any,<:Any,IL}) where IL = IL 

"Determine the needed buffer size for a given Input array and loop ranges lr"
function getbufsize(ia, lr)
    map(mysub(ia,lr.chunks),ia.windows.members) do cr,op
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

"Create buffer for single output"
generate_outbuffer(ia,func,loopranges) = array_from_init(func.init,ia,getbufsize(ia,loopranges))

"Creates buffers for all outputs"
generate_outbuffers(outars,func,loopranges) = generate_outbuffer.(outars,(func,),(loopranges,))

offset_from_range(r) = first(r) .- 1

"Reads data from input array `ia` along domain `r` into `buffer`"
function read_range(r,ia,buffer)
    fill!(buffer,zero(eltype(buffer)))
    mywindowrange = mysub(ia,r)
    inds = map(ia.windows.members,mywindowrange) do w,r
        i = w[r]
        first(first(i)):last(last(i))
    end
    buffer[Base.OneTo.(length.(inds))...] = ia.a[inds...]
    ArrayBuffer(buffer,ia.windows,offset_from_range.(inds),ia.lr)
end

"Wraps output buffer into an ArrayBuffer"
function wrap_outbuffer(r,ia,buffer)
    mywindowrange = mysub(ia,r)
    inds = map(ia.windows.members,mywindowrange) do w,r
        i = w[r]
        first(first(i)):last(last(i))
    end
    ArrayBuffer(buffer,ia.windows,offset_from_range.(inds),ia.lr)
end

function create_buffers(inars, outars, f, loopranges)
    inbuffers_pure = generate_inbuffers(inars,loopranges)
    outbuffers_pure = generate_outbuffers(outars,f,loopranges)
    inbuffers_pure, outbuffers_pure
end