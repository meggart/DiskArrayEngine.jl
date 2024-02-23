
@testset "Identify output chunk overlaps" begin
    import DiskArrayEngine as DAE
    using Zarr

    lr = DAE.ProductArray(([i:i+3 for i in 1:4:28],[i:i for i in 1:3],[i:i+1 for i in 1:2:4]))
    outspecs = DAE.create_outwindows((168,4),dimsmap=(1,3),windows=([i:(i+5) for i in range(1,step=6,length=28)],1:4))

    outar = zzeros(Float32,168,4,chunks = (48,2))
    @test DAE.is_output_chunk_overlap(outspecs,outar,1,lr)
    @test !DAE.is_output_chunk_overlap(outspecs,outar,2,lr)
    @test !DAE.is_output_chunk_overlap(outspecs,outar,3,lr)
    outar = zzeros(Float32,168,4,chunks = (12,1))
    @test !DAE.is_output_chunk_overlap(outspecs,outar,1,lr)
    @test !DAE.is_output_chunk_overlap(outspecs,outar,3,lr)
    outar = zzeros(Float32,168,4,chunks = (24,3))
    @test !DAE.is_output_chunk_overlap(outspecs,outar,1,lr)
    @test DAE.is_output_chunk_overlap(outspecs,outar,3,lr)
end