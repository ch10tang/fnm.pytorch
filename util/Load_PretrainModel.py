
def Load_PretrainModel(model, model_dict, pretrain_model_dict):
    Data = {k: v for k, v in pretrain_model_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(Data)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    return model