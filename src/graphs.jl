using Graphs: Edge, AbstractGraph, Graphs
struct MwopConnection
    inputids
    outputids
    f
    inwindows
    outwindows
end
describe(c::MwopConnection,x) = describe(c.f.f,x)
describe(f::Union{BlockFunction,ElementFunction},_) = string(f.f)
struct MwopOutNode
    ismem
    chunks
    size
    eltype
end
Base.ndims(n::MwopOutNode) = length(n.size)
Base.size(n::MwopOutNode) = n.size
describe(_::MwopOutNode,i) = "Output $i" 
describe(i::InputArray,j) = describe(i.a,j)
describe(z::ZArray,_) = Zarr.zname(z)
describe(z::Array{<:Any,0},_) = z[]
describe(z,i) = "Input $i"
EmptyInput(n::MwopOutNode) = EmptyInput{n.eltype,length(n.size)}(n.size)

mutable struct MwopGraph <: AbstractGraph{Int}
    dims::UnitRange{Int}
    nodes
    connections
end
MwopGraph() = MwopGraph(1:0, [], [])
Base.zero(::Type{MwopGraph}) = MwopGraph()
function Graphs.edges(g::MwopGraph)
    edges = Edge[]
    for c in g.connections
        for from in c.inputids
            for to in c.outputids
                push!(edges,Edge(from,to))
            end
        end
    end
    edges
end
Graphs.edgetype(::MwopGraph) = Edge
function Graphs.has_edge(g::MwopGraph,s,d)
    any(g.connections) do c
        any(c.inputids) do src
            any(c.outputids) do dest
                src==s && dest==d
            end
        end
    end
end
function get_edge(g::MwopGraph,s,d)
    r = map(g.connections) do c
        a = map(c.inputids) do src
            findfirst(c.outputids) do dest
                src==s && dest==d
            end
        end
        iin = findfirst(!isnothing,a)
        isnothing(iin) ? nothing : (iin,a[iin])
    end
    iconn = findfirst(!isnothing,r)
    g.connections[iconn],r[iconn]...

end
Graphs.has_vertex(g::MwopGraph,i) = 0<i<=length(g.nodes)
Graphs.is_directed(::Type{<:MwopGraph}) = true
Graphs.ne(g::MwopGraph) = sum(c->length(c.inputids)*length(c.outputids),g.connections)
Graphs.nv(g::MwopGraph) = length(g.nodes)
Graphs.vertices(g::MwopGraph) = 1:length(g.nodes)
function inconnections(g, v)
    findall(c->in(v,c.outputids),g.connections)
end
function Graphs.inneighbors(g::MwopGraph,v)
    foldl(inconnections(g,v), init=Int[]) do s,ic
        append!(s,g.connections[ic].inputids)
    end
end
function outconnections(g, v)
    findall(c->in(v,c.inputids),g.connections)
end
function Graphs.outneighbors(g::MwopGraph,v)
    foldl(outconnections(g,v), init=Int[]) do s,ic
        append!(s,g.connections[ic].outputids)
    end
end

function edgenames(g::MwopGraph)
    n = String[]
    for c in g.connections
        s = describe(c,1)
        for _ in 1:(length(c.inputids)*length(c.outputids))
            push!(n,s)
        end
    end
    n
end
function nodenames(g::MwopGraph)
    map(i->describe(i[2],i[1]),enumerate(g.nodes))
end

function add_node!(g::MwopGraph, a)
    id = findfirst(Base.Fix1(===,a), g.nodes)
    if id === nothing
        push!(g.nodes, a)
        id = length(g.nodes)
    end
    id
end
add_node!(g::MwopGraph, inar::InputArray) = add_node!(g, inar.a)
add_node!(g::MwopGraph, outspec::NamedTuple,et) = add_node!(g,MwopOutNode(outspec.ismem, outspec.chunks,map(maximum,outspec.lw.windows.members),et))
function add_node!(g::MwopGraph, n::MwopOutNode)
    push!(g.nodes, n)
    length(g.nodes)
end

## Two connections are seen as equivalent when they apply the same operation
## on the same inputs
function equivalent_conection(c1,c2)
    c1.f == c2.f &&
    c1.inputids == c2.inputids &&
    c1.inwindows == c2.inwindows &&
    c1.outwindows == c2.outwindows
end

function merge_connection!(g,i,j)
    #Reroute all outputs of j
    c1 = g.connections[i]
    c2 = g.connections[j]
    deleteat!(g.connections,j)
    for (o_new,o_old) in zip(c1.outputids,c2.outputids)
        for c in g.connections
            ii = findall(==(o_old),c.inputids)
            c.inputids[ii] .= o_new
        end
    end
    for del in c2.outputids
        delete_node!(g,del)
    end
end
function delete_node!(g,del)
    deleteat!(g.nodes,del)
    for c in g.connections
        for i in 1:length(c.inputids)
            if c.inputids[i] > del
                c.inputids[i] = c.inputids[i]-1
            end
        end
        for i in 1:length(c.outputids)
            if c.outputids[i] > del
                c.outputids[i] = c.outputids[i]-1
            end
        end
    end
end

function remove_aliases!(g::MwopGraph)
    for i in 1:length(g.connections)
        for j in (i+1):length(g.connections)
            if equivalent_conection(g.connections[i],g.connections[j])
                merge_connection!(g,i,j)
                return remove_aliases!(g)
            end
        end
    end
end


function to_graph!(g, op::GMDWop, aliases=Dict())
    output_ids = map(enumerate(op.outspecs),op.f.outtype) do (iout,outspec),et
        if iout in keys(aliases)
            aliases[iout]
        else
            add_node!(g, outspec, et)
        end
    end
    input_ids = map(op.inars) do inar
        if inar.a isa GMWOPResult
            r = inar.a
            i = getioutspec(r)
            outspec = getoutspec(r)
            id = add_node!(g,outspec,eltype(inar.a))
            to_graph!(g,r.op,Dict(i=>id))
            id
        else
            add_node!(g, inar)
        end
    end
    inwindows = map(i->i.lw,op.inars)
    outwindows = map(i->i.lw,op.outspecs)
    push!(g.connections,MwopConnection(collect(input_ids),output_ids,op.f,inwindows, outwindows))
    g.dims = 1:length(op.windowsize)
    g
end



using Graphs: SimpleDiGraph, Graphs
struct DimensionGraph
  nodegraph
  dimgraph
  nodes
  concomps
end

function DimensionGraph(g)
  dimnodes = Tuple{Int,Int}[]
  for i in eachindex(g.nodes)
    for d in 1:ndims(g.nodes[i])
      push!(dimnodes,(i,d))
    end
  end
  dimsgraph = SimpleDiGraph(length(dimnodes))
  for conn in g.connections
    for (i_in,inid) in enumerate(conn.inputids)
      li_in = getloopinds(conn.inwindows[i_in])
      for (i_out,outid) in enumerate(conn.outputids)
        li_out = getloopinds(conn.outwindows[i_out])
        matches = find_matches(li_in, li_out)
        for m in matches 
          innode = findfirst(==((inid,first(m))),dimnodes)
          outnode = findfirst(==((outid,last(m))),dimnodes)
          Graphs.add_edge!(dimsgraph,Graphs.Edge(innode,outnode))
        end
      end
    end
  end
  dimcons = Graphs.connected_components(dimsgraph)
  DimensionGraph(g,dimsgraph,dimnodes,dimcons)
end

function find_matches(t1,t2)
  matches = Pair{Int,Int}[]
  cands = union(t1,t2)
  for c in cands
    i1 = findfirst(==(c),t1)
    i2 = findfirst(==(c),t2)
    !isnothing(i1) && !isnothing(i2) && push!(matches,i1=>i2)
  end
  matches
end

function eliminate_node(nodegraph,i_eliminate,strategies,appliedstrat)
    inconids = inconnections(nodegraph,i_eliminate)
    outconids = outconnections(nodegraph,i_eliminate)
    inconns = nodegraph.connections[inconids]
    outconns = nodegraph.connections[outconids]

    inconn = only(inconns)
    outconn = only(outconns)

    dimmap = create_loopdimmap(inconn,outconn,i_eliminate)

    newop = merge_operations(appliedstrat,inconn,outconn,i_eliminate, dimmap)

    newconn = merged_connection(appliedstrat, nodegraph, inconn,outconn,i_eliminate, newop, strategies, dimmap)

    deleteat!(nodegraph.connections,[inconids;outconids])
    push!(nodegraph.connections,newconn)
end

