using Graphs: Edge, AbstractGraph, Graphs
struct MwopConnection
    inputids
    outputids
    f
    inwindows
    outwindows
end
describe(c::MwopConnection,_) = string(c.f.f.f)
struct MwopOutNode
    ismem
    chunks
end
describe(_::MwopOutNode,i) = "Output $i" 
describe(i::InputArray,j) = describe(i.a,j)
describe(z::ZArray,_) = Zarr.zname(z)
describe(z::Array{<:Any,0},_) = z[]
describe(z,i) = "Input $i"

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
add_node!(g::MwopGraph, outspec::NamedTuple) = add_node!(g,MwopOutNode(outspec.ismem, outspec.chunks))
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
    output_ids = map(enumerate(op.outspecs)) do (iout,outspec)
        if iout in keys(aliases)
            aliases[iout]
        else
            add_node!(g, outspec)
        end
    end
    input_ids = map(op.inars) do inar
        if inar.a isa GMWOPResult
            r = inar.a
            i = getioutspec(r)
            outspec = getoutspec(r)
            id = add_node!(g,outspec)
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

