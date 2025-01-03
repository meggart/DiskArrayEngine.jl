const K_RestartHeader = UInt64(19021983)
const K_ProductArray     = UInt8(0)
const K_RegularChunks       = UInt8(1)
const K_IrregularChunks = UInt8(2)

struct Restarter{LRT,R<:Union{Nothing, Vector{LRT}}}
    file::String
    remaining_loopranges::R
    LRT::Type{LRT}
end
Base.ndims(::Restarter{LRT}) where LRT = fieldcount(LRT)

create_restarter(::Nothing, _,_) = nothing

function create_restarter(filename,lr,restartmode)
    filename === nothing && return nothing
    restartmode in (:continue,:overwrite) || error("Unknown restartmode")
    if isfile(filename) && restartmode == :continue
        restarter = Restarter(filename,nothing,eltype(lr))
        loopranges_loaded = orig_loopranges(restarter)
        if loopranges_loaded != lr
            error("Loopranges in file do not match")
        end
        entries = finished_entries(restarter)
        if isempty(entries)
            return restarter
        else
            loopranges_remaining = setdiff(lr,entries)
            return Restarter(filename,loopranges_remaining,eltype(lr))
        end
    else
        open(filename,"w") do f
            write(f,K_RestartHeader)
            iob = IOBuffer()
            putitem(iob,lr)
            alloopranges = take!(iob)
            write(f,UInt64(length(alloopranges)+16))
            write(f,alloopranges)
        end
        Restarter(filename, nothing, eltype(lr))
    end
end

function putitem(f::IO,lr::ProductArray)
    write(f,K_ProductArray)
    write(f,UInt64(length(lr.members)))
    foreach(lr.members) do m
        putitem(f,m)
    end
end
function putitem(f::IO,g::DiskArrays.RegularChunks)
    write(f,K_RegularChunks)
    write(f,g.cs)
    write(f,g.offset)
    write(f,g.s)
end
function putitem(f::IO, g::DiskArrays.IrregularChunks)
    write(f,K_IrregularChunks)
    write(f,UInt64(length(g.offsets)))
    write(f,g.offsets)
end
function putitem(f::IO, i::UnitRange{Int})
    write(f,first(i))
    write(f,last(i))
end

function orig_loopranges(r::Restarter)
    open(r.file,"r") do f
        read(f,UInt64) == K_RestartHeader || error("Not a valid Restart file")
        lheader = read(f,UInt64)
        nextitem = read(f,UInt8)
        if nextitem == K_ProductArray
            n_members = Int(read(f,UInt64))
            members = ntuple(n_members) do _
                readmember(f)
            end
            ProductArray(members)
        else
            error("Unknown Looprange type")
        end
    end
end

function finished_entries(r::Restarter{LRT}) where LRT
    nd = fieldcount(LRT)
    open(r.file,"r") do f
        read(f,UInt64) == K_RestartHeader || error("Not a valid Restart file")
        lheader = read(f,UInt64)
        seek(f,lheader)
        out = LRT[]
        while !eof(f)
            entry = ntuple(nd) do _
                read(f,Int):read(f,Int)
            end
            push!(out,entry)
        end
        out
    end
end

function readmember(f)
    membertype = read(f,UInt8)
    if membertype == K_RegularChunks
        cs, offs, s = read(f,Int), read(f,Int), read(f,Int)
        DiskArrays.RegularChunks(cs,offs,s)
    elseif membertype == K_IrregularChunks
        n = Int(read(f,UInt64))
        r = Vector{Int}(undef,n)
        read!(f,r)
        DiskArrays.IrregularChunks(r)
    else
        error("Unknown membertype")
    end
end



function add_entry(r::Restarter, inow::Tuple)
    open(r.file,"a") do f
        foreach(inow) do ii
            putitem(f,ii)
        end
    end
end