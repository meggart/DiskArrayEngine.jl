using DiskArrayEngine
using Test
using Zarr
import DiskArrayEngine as DAE

@testset "GMWOPResult" begin
    a = reshape(1:20,4,5)
    store = DirectoryStore(tempname())
    za = ZArray(a,store,path="t2m")
    b = ones(4)
    inars = (DAE.InputArray(za,windows=([1:2,3:4],1:5)), DAE.InputArray(b,windows=([1:2,3:4],)))
    outwindows = (DAE.create_outwindows((2,5),windows = (1:2,1:5)),DAE.create_outwindows((4,5),windows=([1:2,3:4],1:5)))
    f = DAE.create_userfunction((Int64,Int64),is_mutating=false) do x,y
        sum(x)+sum(y),x.+y
    end
    op = DAE.GMDWop(inars,outwindows, f)
    res1, res2 = DAE.results_as_diskarrays(op)
    @test res1 isa DAE.GMWOPResult
    @test DAE.getoutspec(res1) == outwindows[1]
    @test DAE.getioutspec(res1) == 1
    @test size(res1) == (2,5)
    @test res1[1,1] == 5
    @test res1[:,:] == [5 13 21 29 37; 9 17 25 33 41]
    buf = IOBuffer()
    show(buf,MIME"text/plain"(),res1)
    r = String(take!(buf))
    @test r == "Output #1: 2 x 5 GMWOPResult{Int64}\nInputs:    4 x 5 Zarray{Int64} \"t2m\"4 Vector{Float64} Input #2"
    @test res2 isa DAE.GMWOPResult
    @test DAE.getoutspec(res2) == outwindows[2]
    @test DAE.getioutspec(res2) == 2
    @test size(res2) == (4,5)
    @test res2[1,1] == 2
    buf = IOBuffer()
    show(buf,MIME"text/plain"(),res2)
    r = String(take!(buf))
    @test r == "Output #2: 4 x 5 GMWOPResult{Int64}\nInputs:    4 x 5 Zarray{Int64} \"t2m\"4 Vector{Float64} Input #2"
end
