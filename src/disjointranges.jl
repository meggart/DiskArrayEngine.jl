function rangelt(x, y)
    x1, x2 = extrema(x)
    y1, y2 = extrema(y)
    lowerlt = x1 < y1
    upperlt = x2 < y2
    lowerlet = x1 <= y1
    upperlet = x2 <= y2
    if lowerlt
        upperlet || error("Ranges contain each other")
        return true
    end
    if upperlt
        lowerlet || error("Ranges contain each other")
        return true
    end
    return false
end

windowminimum(w::AbstractVector{<:Number}) = minimum(w)
windowminimum(w) = minimum(windowmin, w)

windowmaximum(w::AbstractVector{<:Number}) = maximum(w)
windowmaximum(w) = maximum(windowmax, w)

last_contains_value(w::AbstractVector{<:Number}, i) = findlast(<=(i), w)
last_contains_value(w::AbstractRange, i) = searchsortedlast(w, i)
function last_contains_value(w, i)
    ii = findlast(r -> i in inner_range(r), w)
    if ii === nothing
        length(w) + 1
    else
        ii
    end
end
