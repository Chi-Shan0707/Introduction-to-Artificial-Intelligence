optimizer = torch.optim.SGD(model.parameters(),lr=0.01)


for epoch in range(10):
    outputs = model(x)

    loss =criterion(outputs, y_clas)

    optimizer.zero_grad()


    loss.backward()


    optimizer.step()





    