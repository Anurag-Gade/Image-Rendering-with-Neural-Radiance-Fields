from utils import *
from encoder_params import *
from train_params import *
from model import *
from generator import *
from display import *

import_libraries()



model = NERF_Model(10,256,5,sine_activation=True,positional_encoding=False)
x = torch.randn([10,2])
model(x).shape

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=alpha,momentum=momentum) 

g = generator(32)
image_size = 100
for epoch in tqdm.tqdm(range(epochs)):
    losses =0
    model.to(device)
    for i in range(image_size**2//batch_size):
        x,y = next(g)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        y1 = model(x)
        loss = loss_function(y1,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses=loss.cpu().detach().numpy()

    if epoch%50==0:
        display(model.cpu() )
    print(f"Epoch : {epoch},loss :{losses} ")#
