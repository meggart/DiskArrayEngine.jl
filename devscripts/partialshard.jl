using Dagger


struct PartialShard
    chunks::Dict{Dagger.Processor,Tuple{Dagger.Chunk,Dagger.Chunk}}
end


function partialshard(@nospecialize(f); procs=nothing, workers=nothing, per_thread=false)
    if procs === nothing
        if workers !== nothing
            procs = [OSProc(w) for w in workers]
        else
            procs = lock(Dagger.Sch.eager_context()) do
                copy(Dagger.Sch.eager_context().procs)
            end
        end
        if per_thread
            _procs = Dagger.ThreadProc[]
            for p in procs
                append!(_procs, filter(p->p isa Dagger.ThreadProc, Dagger.get_processors(p)))
            end
            procs = _procs
        end
    else
        if workers !== nothing
            throw(ArgumentError("Cannot combine `procs` and `workers`"))
        elseif per_thread
            throw(ArgumentError("Cannot combine `procs` and `per_thread=true`"))
        end
    end
    isempty(procs) && throw(ArgumentError("Cannot create empty Shard"))
    shard_dict = Dict{Dagger.Processor,Tuple{Dagger.Chunk,Dagger.Chunk}}()
    for proc in procs
        scope = proc isa OSProc ? ProcessScope(proc) : ExactScope(proc)
        thunk = Dagger.@spawn scope=scope Dagger._mutable_inner(f, proc, scope)
        uv = Dagger.@mutable Ref(false)
        shard_dict[proc] = (uv,fetch(thunk)[])
    end
    return PartialShard(shard_dict)
end

function Dagger.move(from_proc::Dagger.Processor, to_proc::Dagger.Processor, shard::PartialShard)
    # Match either this proc or some ancestor
    # N.B. This behavior may bypass the piece's scope restriction
    proc = to_proc
    if haskey(shard.chunks, proc)
        a,b = shard.chunks[proc]
        Dagger.spawn(a) do aa 
            aa[]=true
        end
        return Dagger.move(from_proc, to_proc, b)
    end
    parent = Dagger.get_parent(proc)
    while parent != proc
        proc = parent
        parent = Dagger.get_parent(proc)
        if haskey(shard.chunks, proc)
            a,b = shard.chunks[proc]
            Dagger.spawn(a) do aa 
                aa[]=true
            end
            return Dagger.move(from_proc, to_proc, b)
        end
    end

    throw(KeyError(to_proc))
end
getiter(s::PartialShard) = Iterators.map(last,Iterators.filter(i->fetch(first(i))[],values(s.chunks)))
Base.iterate(s::PartialShard) = iterate(getiter(s))
Base.iterate(s::PartialShard, state) = iterate(getiter(s), state)
Base.length(s::PartialShard) = sum(i->fetch(first(i))[],values(s.chunks))