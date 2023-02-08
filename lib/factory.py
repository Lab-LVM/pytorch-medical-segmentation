import torch
import torch.nn as nn
import numpy as np
from lib.networks import TransCASCADE, PVT_CASCADE
from lib.cnn_vit_backbone import TransUnet
from lib.swinunet import SwinUnet
from lib.swin_cfg import get_config

def create_model(args, config_vit=None):
    if args.model.startswith("TransUnet"):
        config_vit.encoderType = "normal"
        config_vit.img_size = args.img_size
        args.base_lr = 0.01
        model = TransUnet(config_vit).cuda()
        if args.model.split("-")[-1] == "pt":
            print("## Loading Weight from", config_vit.pretrained_path, "...")
            model.load_from(weights=np.load(config_vit.pretrained_path))
            print("## Successfully Loading!!")

    elif args.model.startswith("SwinUnet"):
        config = get_config(args)
        args.base_lr = 0.05
        model = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
        # if args.model.split("-")[-1] == "pt":
        print("## Loading Weight from", config.MODEL.PRETRAIN_CKPT, "...")
        model.load_from(config)
        print("## Successfully Loading!!")

    elif args.model.startswith("TransCASCADE"):
        config_vit.encoderType = "normal"
        model = TransCASCADE(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda() # model initialization for TransCASCADE
        if args.model.split("-")[-1] == "pt":
            print("## Loading Weight from",config_vit.pretrained_path,"...")
            model.load_from(weights=np.load(config_vit.pretrained_path))
            print("## Sucessfully Loading")

    elif args.model.startswith("PVT-CASCADE"):
        if config_vit==None:
            model = PVT_CASCADE().cuda()
        else:
            config_vit.encoderType = "normal"
            model = PVT_CASCADE(n_class=config_vit.n_classes).cuda() # model initialization for PVT-CASCADE. comment above two lines if use PVT-CASCADE
        if args.model.split("-")[-1] == "pt":
            print("Load weights...")
            weight = torch.load("pretrained_pth/pvt/pvt_v2_b2.pth")
            model.load_pvt_weights_from(weight)
            print("Successfully load weights!!!")

    elif args.model.startswith("multiway"):
        config_vit.encoderType = "multiway"
        model_cfg = args.model.split("-")
        config_vit.img_size = args.img_size
        config_vit.hidden_size = int(model_cfg[2][1:])  #d192
        config_vit.direction = int(model_cfg[3][1:])    #d0,1,2
        config_vit.num_path = int(model_cfg[4][1:])     #p3,2,1
        config_vit.concat = True if model_cfg[5]=="concat" else False
        config_vit.pixshuf_factor = 2
        if model_cfg[1] == "TransUnet":
            model = TransUnet(config_vit).cuda()
        elif model_cfg[1] == "TransCASCADE":
            model = TransCASCADE(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda() # model initialization for TransCASCADE
        else:
            # have to implement multiway-PVT-CASCADE
            assert False, f"doesn't exist \"{model_cfg[1]}\" model"
        

    else:
        assert False, f"{args.model} is not supported yet"
    print('#params: {:,}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # model parellel    
    if len(args.cuda) > 1:
        model = nn.DataParallel(model)
    return model