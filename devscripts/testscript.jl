using Dagger

example_data = [
  [
    [:a=>1, :b=>2, :b=>3],
    [:b=>3, :b=>2, :a=>1],
  ], [
    [:a=>1, :c=>10, :d=>-1, :c=>10, :d=>-1],
    [:c=>11, :d=>-2, :d=>-2, :c=>11],
  ], [
    [:e=>3, :e=>3],
    [:e=>4, :e=>4, :a=>1],
  ]
]

#Kepp track of sum and count
function accumulate_data(x,agg) 
  for (name,val) in x
    n,s = get!(agg,name,(0,0))
    agg[name] = (n+1,s+val)
  end
  nothing
end


function merge_and_flush_outputs(aggregator)
  if isempty(aggregator)
    return nothing
  else
  merged_aggregator = fetch(reduce(aggregator) do d1, d2
    merge(fetch(d1),fetch(d2)) do (n1,s1),(n2,s2)
      n1+n2,s1+s2
    end
  end)
  for k in keys(merged_aggregator)
    n,s = merged_aggregator[k]
    if n==4
      @info "$k: $s"
      delete!(merged_aggregator,k)
    end
  end
  merged_aggregator
  end
end

function filter_aggregator(aggregator,used_procs)
  return (v for (k,v) in aggregator.chunks if k in used_procs)
end
function filter_aggregator(aggregator,::Nothing)
  values(aggregator.chunks)
end
function merge_local_outputs(aggregator,procs)
  aggregator_copies = Dagger.spawn_bulk() do
    map(filter_aggregator(aggregator,procs)) do agg
      Dagger.spawn(copy,agg)
    end
  end
  # Merge and flush all aggregator copies
  Dagger.@spawn merge_and_flush_outputs(aggregator_copies)
end


aggregator = Dagger.shard(;per_thread=true) do 
  Dict{Symbol,Tuple{Int,Int}}()
end;
r = map(example_data) do group
  Dagger.spawn(group) do group
    Dagger.spawn_sequential() do
      localaggregator = Dagger.shard(;per_thread=true) do
        Dict{Symbol,Tuple{Int,Int}}()
      end
      r = Dagger.spawn_bulk() do
        map(group) do subgroup
          Dagger.spawn(accumulate_data,subgroup,localaggregator)
        end
      end
      procs = Dagger.processor.(fetch.(r,raw=true))
      unflushed_data = merge_local_outputs(localaggregator,procs)
      wait(Dagger.spawn(unflushed_data,aggregator) do rem_data, agg
        merge!(agg,rem_data) do (n1,s1),(n2,s2)
          n1+n2,s1+s2
        end
      end)
    end
    true
  end
end;
fetch.(r)
wait(merge_local_outputs(aggregator,nothing));
