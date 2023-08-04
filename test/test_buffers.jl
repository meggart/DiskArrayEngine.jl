@testset "Computation of output buffer repeats" begin
    
    lr = ProductArray(([i:i+4 for i in 1:5:20],[i:i for i in 1:3],[i:i+1 for i in 1:2:4]))
    lw = LoopWindows(ProductArray((DiskArrayEngine.to_window([1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7]),1:4)),Val((1,3)))
    ind = lr[1,2,2]
    @test DiskArrayEngine.bufferrepeat(lr[1,2,2],lr,lw) == 6
    @test DiskArrayEngine.bufferrepeat(lr[2,2,2],lr,lw) == 9
    @test DiskArrayEngine.bufferrepeat(lr[4,2,2],lr,lw) == 3

end