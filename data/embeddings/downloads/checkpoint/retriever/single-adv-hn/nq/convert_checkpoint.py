import torch
x = torch.load('bert-base-encoder.cp', map_location=torch.device('cpu'))
x['epoch'] = 0
x['offset'] = 0
x['scheduler_dict'] = None
x['optimizer_dict']['param_groups'][0]['lr'] = 2e-5
x['optimizer_dict']['param_groups'][1]['lr'] = 2e-5
torch.save(x, 'bert-base-encoder_no_scheduler_dict.cp')
