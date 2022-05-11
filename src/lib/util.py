
def to_device(tup, device):
    on_device = []
    for i in range(len(tup)):
        if type(tup[i]) == list:
            on_device.append(to_device(tup[i], device))
        else:
            tensor = tup[i].to(device)
            on_device.append(tensor)
    return on_device