#Implement the window trait for a few types. Main info to be known
#is if it contains ranges or single numbers, if windows are monotonic
#and if they are overlapping


struct Increasing end
struct Decreasing end
struct CyclicIncreasing end
struct Unordered end
 
struct Sparse 
    sparsity::Float64
end
struct Dense end

struct Overlapping end
struct Repeating
    nrep::Float64
end
struct NonOverlapping end

struct Window{T,W<:AbstractVector{T},O,L,S} <: AbstractVector{T}
    w::W
    ordering::O
    overlap::L
    sparsity::S
end
to_window(w::Window) = w
Base.size(w::Window,i...) = size(w.w,i...)
Base.getindex(w::Window,i) = w.w[i]
get_ordering(w::Window) = w.ordering
get_overlap(w::Window) = w.overlap
get_sparsity(w::Window) = w.sparsity
inner_index(w::Window, i) = inner_index(w.w, i)
inner_index(w, i) = w[i]
purify_window(w) = w
purify_window(w::Window) = w.w

"A group of window steps merged together in a block operation"
struct WindowGroup{P}
  parent::P
  g::UnitRange{Int}
  function WindowGroup(parent, g::UnitRange{Int})
    if eltype(parent) <: WindowGroup
      throw(ArgumentError("WindowGroup cannot contain another WindowGroup"))
    end
    if !(0 < first(g) <= last(g) <= length(parent))
        @show first(g), last(g), length(parent)
        error()
    end
    new{typeof(parent)}(parent, g)
  end
end
inner_range(w::WindowGroup) = first(inner_range(w.parent[first(w.g)])):last(inner_range(w.parent[last(w.g)]))
inner_range(i::Number) = i:i
inner_range(i::AbstractRange) = i
inner_values(w::WindowGroup) = w.parent[w.g]
inner_values(i) = i
inner_getindex(w::Vector{<:WindowGroup}, i::Int64) = first(w).parent[i]
compute_ordering(r::AbstractVector{<:WindowGroup}) = compute_ordering(first(r).parent)
compute_overlap(r::AbstractVector{<:WindowGroup}, ordering) = compute_overlap(inner_range.(r), ordering)
compute_sparsity(r::AbstractVector{<:WindowGroup}) = compute_sparsity(first(r).parent)
windowmax(r) = maximum(r)
windowmin(r) = minimum(r)
windowmax(r::WindowGroup) = maximum(i -> windowmax(r.parent[i]), r.g)
windowmin(r::WindowGroup) = minimum(i -> windowmin(r.parent[i]), r.g)

function compute_ordering(r)
    exts = extrema.(r)
    allsorted(x;rev=false) = issorted(x,by=first;rev) && issorted(x,by=last;rev)
    ordering = if allsorted(exts)
        Increasing()
    elseif allsorted(exts,rev=true)
        Decreasing()
    else
        Unordered()
    end
end




switchfunc(::Decreasing,x,y) = y,x
switchfunc(::Increasing,x,y) = x,y

function lt_range(x,y,lt)
    if lt(first(x),first(y))
        lt(last(y),last(x)) && throw(ArgumentError("Window a contains b"))
        true
    elseif lt(last(x),last(y))
        lt(first(y),first(x)) && throw(ArgumentError("Window a contains b"))
        true
    else
        false
    end
end

function compute_overlap(r,ordering)
    f, r2 = Iterators.peel(r)
    init = (nequal=0,noverlap=0,lastr=f)
    rover = foldl(r2,init=init) do old,new
        (;nequal,noverlap,lastr) = old
        i1,i2 = switchfunc(ordering,lastr,new)
        if lastr == new
            nequal += 1
        elseif last(i1) >= first(i2)
            noverlap +=1
        end
        (;lastr=new,nequal,noverlap)
    end
    if rover.noverlap > 0
        return Overlapping()
    elseif rover.nequal > 0
        nnonrep = length(r)-rover.nequal
        return Repeating(rover.nequal/nnonrep)
    else
        return NonOverlapping()
    end
end


function compute_sparsity(r)
    total_range = mapreduce(maximum,max,r) - mapreduce(minimum,min,r)
    covered_range = mapreduce(length,+,r)
    if covered_range / total_range < 0.5
        return Sparse(covered_range/total_range)
    else
        return Dense()
    end
end


function to_window(r)
    eltype(r) <: Int || eltype(r) <: AbstractUnitRange{Int} || eltype(r) <: WindowGroup || throw(ArgumentError("Windows must contain Ints or UnitRanges as elements"))
    ordering = compute_ordering(r)
    overlap = compute_overlap(r,ordering)
    sparsity = compute_sparsity(r)
    Window(r,ordering,overlap,sparsity)
end

struct MovingWindow{C} <: AbstractVector{UnitRange{Int}}
    first::Int
    steps::Int
    width::Int
    n::Int
    clamp::C
end
MovingWindow(first, steps, width, n) = MovingWindow(first, steps, width, n, nothing)

Base.size(m::MovingWindow) = (m.n,)
Base.getindex(m::MovingWindow{Nothing}, i::Int) = (m.first+(i-1)*m.steps):(m.first+(i-1)*m.steps+m.width-1)
Base.getindex(m::MovingWindow, i::Int) = max(first(m.clamp), m.first + (i - 1) * m.steps):min(last(m.clamp), m.first + (i - 1) * m.steps + m.width - 1)
get_ordering(w::MovingWindow) = w.steps > 0 ? Increasing() : Decreasing()
get_overlap(w::MovingWindow) = w.width > abs(w.steps) ? Overlapping() : NonOverlapping()
get_sparsity(w::MovingWindow) = abs(w.steps) > 2*w.width ? Sparse(w.width/abs(w.steps)) : Dense()
to_window(r::MovingWindow) = r



get_ordering(r::AbstractRange{Int}) = step(r)>0 ? Increasing() : Decreasing()
get_overlap(r::AbstractRange{Int}) = NonOverlapping()
get_sparsity(r::AbstractRange{Int}) = abs(step(r)) > 2 ? Sparse(1/abs(step(r))) : Dense()
to_window(r::AbstractRange{Int}) = r

struct Repeated <: AbstractVector{Int}
    val::Int
    length::Int
end
Base.size(r::Repeated) = (r.length,)
Base.IndexStyle(::Type{Repeated}) = Base.IndexLinear()
Base.getindex(r::Repeated, ::Int) = r.val
get_ordering(::Repeated) = Increasing()
get_overlap(r::Repeated) = Repeating(Float64(r.length))
get_sparsity(r::Repeated) = Dense()
to_window(r::Repeated) = r
