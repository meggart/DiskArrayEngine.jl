function rangelt(x,y)
    x1,x2 = extrema(x)
    y1,y2 = extrema(y)
    lowerlt = x1<y1
    upperlt = x2<y2
    lowerlet = x1<=y1
    upperlet = x2<=y2
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
windowminimum(w) = minimum(minimum,w)

windowmaximum(w::AbstractVector{<:Number}) = maximum(w)
windowmaximum(w) = maximum(maximum,w)

last_contains_value(w::AbstractVector{<:Number},i) = findlast(<=(i),w)
last_contains_value(w::AbstractRange,i) = searchsortedlast(w,i)
last_contains_value(w,i) = findlast(r->maximum(r)<=i,w)