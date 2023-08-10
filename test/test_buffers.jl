@testset "Output buffer collections" begin
  using DiskArrayEngine
  using DiskArrayEngine: ProductArray, LoopWindows, to_window, generate_outbuffer_collection, extract_outbuffer, put_buffer

  lr = ProductArray(([1:2,3:3,4:9,10:15,16:20],[i:i for i in 1:3],[i:i+1 for i in 1:2:4]))
  outspecs = create_outwindows((7,4),dimsmap=(1,3),windows=([1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7],1:4))

  @test DiskArrayEngine.bufferrepeat(lr[1,2,2],lr,outspecs.lw) == 6
  @test DiskArrayEngine.bufferrepeat(lr[2,2,2],lr,outspecs.lw) == 6
  @test DiskArrayEngine.bufferrepeat(lr[3,2,2],lr,outspecs.lw) == 3


  buftype = Int32
  init = Int32(0)
  obc = generate_outbuffer_collection(outspecs,buftype,lr)
  bufview1 = extract_outbuffer(lr[1,1,1],lr,outspecs,init,buftype,obc)
  @test length(obc.buffers) == 1
  k1 = DiskArrayEngine.BufferIndex((1:1,1:2))
  @test haskey(obc.buffers,k1)
  n,ntot,b = obc.buffers[k1]
  @test n[] == 1
  @test ntot == 6
  @test b isa DiskArrayEngine.ArrayBuffer
  @test DiskArrayEngine.getloopinds(b) == (1,3)
  @test b.offsets == (0,0)
  @test b.a == zeros(Int32,2,2)
  @test bufview1 === b
  bufview2 = extract_outbuffer(lr[2,1,1],lr,outspecs,init,buftype,obc)
  n,ntot,b2 = obc.buffers[k1]
  @test n[] == 2
  @test ntot == 6
  @test b2 === b
  @test length(obc.buffers) == 1
  @test bufview2 === b

  outar = zeros(Int32,7,4)
  b.a .= 1
  @test put_buffer(lr[1,1,1], identity, bufview1, obc, outar, nothing) == false
  @test put_buffer(lr[2,1,1], identity, bufview1, obc, outar, nothing) == false
  n[] = 6
  @test put_buffer(lr[2,3,1], identity, bufview1, obc, outar, nothing) == true
  obc.buffers
  @test length(obc.buffers) == 0
  @test outar[1,:] == [1,1,0,0]
  @test all(iszero,outar[2:end,:])

  bufview2 = extract_outbuffer(lr[3,1,1],lr,outspecs,init,buftype,obc)
  k2 = DiskArrayEngine.BufferIndex((2:3,1:2))
  n,ntot,b = obc.buffers[k2]
  @test n[]==1
  @test ntot==3
  b.a .= 2
  n[] = 4
  @test_throws Exception put_buffer(lr[3,3,1], identity, bufview2, obc, outar, nothing)
  n[] = 3
  @test put_buffer(lr[3,3,1], identity, bufview2, obc, outar, nothing) == true
  @test all(==(2),outar[2:3,1:2])
end

@testset "Merging Buffer collection" begin
  using DiskArrayEngine: ProductArray, LoopWindows, to_window, generate_outbuffer_collection, extract_outbuffer, put_buffer,merge_outbuffer_collection

lr = ProductArray(([1:2,3:3,4:9,10:15,16:20],[i:i for i in 1:3],[i:i+1 for i in 1:2:4]))
outspecs = create_outwindows((7,4),dimsmap=(1,3),windows=([1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7],1:4))

buftype = Int32
init = Int32(0)
collections = map(1:3) do i
  obc = generate_outbuffer_collection(outspecs,buftype,lr)
  bufview = extract_outbuffer(lr[3,i,1],lr,outspecs,init,buftype,obc)
  bufview.a.=i
  (obc,)
end
r, = DiskArrayEngine.merge_all_outbuffers(collections,+)
@test haskey(r.buffers,DiskArrayEngine.BufferIndex((2:3,1:2)))
n,ntot,b = r.buffers[DiskArrayEngine.BufferIndex((2:3,1:2))]
@test n[] == 3
@test ntot == 3
outar = zeros(Int32,7,4)
DiskArrayEngine.flush_all_outbuffers((r,),identity,(outar,),nothing)
@test isempty(r.buffers)
@test all(==(6),outar[2:3,1:2])
end