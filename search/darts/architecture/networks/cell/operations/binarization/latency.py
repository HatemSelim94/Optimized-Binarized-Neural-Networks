import torch 

def get_latency(model, input, device=torch.device('cuda'), runs=500):
    torch.backends.cudnn.benchmark = True
    temp = torch.rand([1,3,224,224], device= device) # 1,3,image_size*image_size
    temp_layer = torch.nn.Conv2d(3,3,3, padding=1, bias=False).to(device)
    clonned_inp = input.clone().detach()
    model = model.cuda().eval()
    start = torch.cuda.Event(enable_timing=True) # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
    end = torch.cuda.Event(enable_timing=True)
    
    #print('GPU warm-up')
    # GPU warm-up (some gpu ops using random input)
    for _ in range(1000):
        with torch.no_grad():
            output = temp_layer(temp)
            #predictions = torch.argmax(output, dim=1)
    #print('Latency test started')
    avg_latency_ms = 0

    # start benchmarking using real data
    for _ in range(runs):
        torch.cuda.synchronize()
        with torch.no_grad():
            start.record()
            output = model(clonned_inp)
            #predictions = torch.argmax(output, dim=1)
            end.record()
            torch.cuda.synchronize()
            avg_latency_ms += start.elapsed_time(end) # the returned time is in ms
    avg_latency_ms = avg_latency_ms/(runs) # in milliseconds
    avg_latency_s = avg_latency_ms / 1000 # in seconds
    #print(f'Latency: {avg_latency_s:0.5f} sec\nFPS: {1/avg_latency_s:0.2f}\nImage size: {img_size}')
    torch.backends.cudnn.benchmark = False
    return avg_latency_ms