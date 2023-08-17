using Dagger
struct LazyShard{P,F}
    chunks::Dict{Dagger.Processor,Dagger.Chunk}
    f::F
    processortype::P
end


function lazyshard(f; per_thread=false)
    pt = per_thread ? Dagger.ThreadProc : OSProc
    shard_dict = Dict{Dagger.Processor,Dagger.Chunk}()
    return LazyShard(shard_dict,f,pt)
end

function Dagger.move(from_proc::Dagger.Processor, to_proc::Dagger.Processor, shard::LazyShard) 
  # Match either this proc or some ancestor
  # N.B. This behavior may bypass the piece's scope restriction
  proc = to_proc
  # scope = P<:OSProc ? ProcessScope(proc) : ExactScope(proc)
  # thunk = Dagger.@spawn scope=scope Dagger._mutable_inner(shard.f, proc, scope)
  if shard.processortype <: Dagger.ThreadProc
    chunk = if !haskey(shard.chunks,proc)
      scope = ExactScope(proc)
      thunk = Dagger.@spawn scope=scope Dagger._mutable_inner(shard.f, proc, scope)
      chunk = fetch(thunk)[]
    else
      shard.chunks[proc]  
    end
    
    return Dagger.move(from_proc, to_proc, chunk)
  else
    parent = proc
    while !(parent <: OSProc)
        parent = Dagger.get_parent(parent)
    end
    proc = parent
    chunk = get!(shard.chunks,proc) do
      scope = ProcessScope(proc)
      thunk = Dagger.@spawn scope=scope Dagger._mutable_inner(shard.f, proc, scope)
      fetch(thunk)[]
    end
    return Dagger.move(from_proc,to_proc, chunk)
  end
end

Base.iterate(s::LazyShard) = iterate(values(s.chunks))
Base.iterate(s::LazyShard, state) = iterate(values(s.chunks), state)
Base.length(s::LazyShard) = length(s.chunks)