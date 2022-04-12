import torch

def get_latency(model, input_shape, device=torch.device('cuda'), runs=500, verbose=False, kd=False):
    torch.backends.cudnn.benchmark = True
    c = input_shape[1]
    temp = torch.randn(input_shape, device= device) # 1,3,image_size*image_size
    temp_layer = torch.nn.Conv2d(c,10,3, padding=1, bias=False).to(device)
    model = model.cuda().eval()
    start = torch.cuda.Event(enable_timing=True) # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
    end = torch.cuda.Event(enable_timing=True)
    
    # GPU warm-up 
    for _ in range(runs):
        with torch.no_grad():
            output = model(temp)
            #predictions = torch.argmax(output, dim=1)
    #print('Latency test started')
    avg_latency_ms = 0
    if kd:
        # start
        for _ in range(runs):
            torch.cuda.synchronize()
            with torch.no_grad():
                start.record()
                output,_ = model(temp)
                predictions = torch.softmax(output, dim=1)
                predictions = torch.argmax(predictions, dim=1)
                end.record()
                torch.cuda.synchronize()
                avg_latency_ms += start.elapsed_time(end) # the returned time is in ms
        avg_latency_ms = avg_latency_ms/(runs) # in milliseconds
        avg_latency_s = avg_latency_ms / 1000 # in seconds
        fps = round(1/avg_latency_s,2)
        #print(f'Latency: {avg_latency_s:0.5f} sec\nFPS: {1/avg_latency_s:0.2f}\nImage size: {img_size}')
        torch.backends.cudnn.benchmark = False
        if verbose:
            print(f'Model latency (ms): {round(avg_latency_ms,2)}')
            print(f'Model FPS: {round(fps,2)}')
        return round(avg_latency_ms,2), fps
    else:
        # start
        for _ in range(runs):
            torch.cuda.synchronize()
            with torch.no_grad():
                start.record()
                output = model(temp)
                predictions = torch.softmax(output, dim=1)
                predictions = torch.argmax(predictions, dim=1)
                end.record()
                torch.cuda.synchronize()
                avg_latency_ms += start.elapsed_time(end) # the returned time is in ms
        avg_latency_ms = avg_latency_ms/(runs) # in milliseconds
        avg_latency_s = avg_latency_ms / 1000 # in seconds
        fps = round(1/avg_latency_s,2)
        #print(f'Latency: {avg_latency_s:0.5f} sec\nFPS: {1/avg_latency_s:0.2f}\nImage size: {img_size}')
        torch.backends.cudnn.benchmark = False
        if verbose:
            print(f'Model latency (ms): {round(avg_latency_ms,2)}')
            print(f'Model FPS: {round(fps,2)}')
        return round(avg_latency_ms,2), fps


def get_latency_hook(model, input_shape, device=torch.device('cuda'), runs=500, verbose=False):
    torch.backends.cudnn.benchmark = True
    c = input_shape[1]
    temp = torch.randn(input_shape, device= device) # 1,3,image_size*image_size
    temp_layer = torch.nn.Conv2d(c,10,3, padding=1, bias=False).to(device)
    model = model.cuda().eval()
    start = torch.cuda.Event(enable_timing=True) # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
    end = torch.cuda.Event(enable_timing=True)
    
    # GPU warm-up 
    for _ in range(runs):
        with torch.no_grad():
            output = model(temp)
            #predictions = torch.argmax(output, dim=1)
    #print('Latency test started')
    avg_latency_ms = 0

    # start
    for _ in range(runs):
        torch.cuda.synchronize()
        with torch.no_grad():
            start.record()
            output = model(temp)
            end.record()
            torch.cuda.synchronize()
            avg_latency_ms += start.elapsed_time(end) # the returned time is in ms
    avg_latency_ms = avg_latency_ms/(runs) # in milliseconds
    avg_latency_s = avg_latency_ms / 1000 # in seconds
    fps = round(1/avg_latency_s,2)
    #print(f'Latency: {avg_latency_s:0.5f} sec\nFPS: {1/avg_latency_s:0.2f}\nImage size: {img_size}')
    torch.backends.cudnn.benchmark = False
    if verbose:
        print(f'Model latency (ms): {round(avg_latency_ms,2)}')
        print(f'Model FPS: {round(fps,2)}')
    return round(avg_latency_ms,2), fps

def latency_forward_hook(mod, input, output):
    shape = input[0].shape
    new_mod = mod.model(mod.get_config())
    latency, _ = get_latency_hook(new_mod, shape, runs=400, verbose=False)
    mod.__total_latency = round(latency, 2)

def calculate_ops_latency(net, input_shape):
    handles = []
    def recurs(net):
        for mod in net.children():
            if hasattr(mod, 'edge_layer'):
                #print(mod)
                handles.append(mod.register_forward_hook(latency_forward_hook))
            else:
                recurs(mod)
    recurs(net)
    dummy_input = torch.randn(input_shape, device='cuda')
    with torch.no_grad():
        net(dummy_input)
    for handle in handles:
        handle.remove()

