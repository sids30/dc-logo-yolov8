Last login: Wed Dec  6 23:12:18 on ttys000
mainfolder-Laptop ~ % python3
Python 3.9.6 (default, Oct  1 2023, 17:38:10)
[Clang 15.0.0 (clang-1500.0.40.1)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from ultralytics import YOLO
>>> model=YOLO("yolov8n.pt")
>>> model.train(data="/Users/mainfolder/data.yaml", epochs=300)
Ultralytics YOLOv8.0.223 🚀 Python-3.9.6 torch-2.1.1 CPU (Apple M2)
engine/trainer: task=detect, mode=train, model=yolov8n.pt, data=/Users/mainfolder/data.yaml, epochs=300, patience=50, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=train11, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, tracker=botsort.yaml, save_dir=/Users/mainfolder/Yolo_Label/runs/detect/train11
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]           
Model summary: 225 layers, 3011043 parameters, 3011027 gradients, 8.2 GFLOPs

Transferred 319/355 items from pretrained weights
TensorBoard: Start with 'tensorboard --logdir /Users/mainfolder/Yolo_Label/runs/detect/train11', view at http://localhost:6006/
Freezing layer 'model.22.dfl.conv.weight'
train: Scanning /Users/mainfolder/YoloDataset/durham_images/Train/Images.
train: New cache created: /Users/mainfolder/YoloDataset/durham_images/Train/Images.cache
val: Scanning /Users/mainfolder/YoloDataset/durham_images/Valid/Images...
val: New cache created: /Users/mainfolder/YoloDataset/durham_images/Valid/Images.cache
Plotting labels to /Users/mainfolder/Yolo_Label/runs/detect/train11/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.002, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 0 dataloader workers
Logging results to /Users/mainfolder/Yolo_Label/runs/detect/train11
Starting training for 300 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/300         0G      1.923      3.365       1.82         11        640: 1
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.60s/it]
                   all         69         74      0.003      0.838      0.198     0.0724

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/300         0G      1.833      2.491      1.696          8        640: 100%|██████████| 13/13 [01:00<00:00,  4.67s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.59s/it]
                   all         69         74    0.00309      0.865      0.201        0.1

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/300         0G        1.7      2.346      1.666          4        640: 100%|██████████| 13/13 [01:01<00:00,  4.70s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.65s/it]
                   all         69         74      0.803      0.221      0.331       0.17

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/300         0G      1.833      2.274      1.666         10        640: 100%|██████████| 13/13 [01:00<00:00,  4.67s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.62s/it]
                   all         69         74      0.557      0.135      0.281      0.103

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/300         0G       1.86      2.157      1.689          7        640: 100%|██████████| 13/13 [01:00<00:00,  4.66s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.62s/it]
                   all         69         74      0.331       0.23      0.238     0.0906

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/300         0G       1.84      2.157      1.651          9        640: 100%|██████████| 13/13 [01:00<00:00,  4.67s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.66s/it]
                   all         69         74      0.364      0.189      0.219     0.0934

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/300         0G      1.855      2.043      1.698          9        640: 100%|██████████| 13/13 [01:00<00:00,  4.68s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.61s/it]
                   all         69         74      0.541      0.162      0.179       0.06

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/300         0G      1.883      2.113      1.717         11        640: 100%|██████████| 13/13 [01:00<00:00,  4.65s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.59s/it]
                   all         69         74      0.144       0.23     0.0828      0.035

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/300         0G      1.976      2.014      1.827         10        640: 100%|██████████| 13/13 [01:00<00:00,  4.66s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74      0.168      0.212      0.133     0.0524

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/300         0G      1.863      1.979      1.747         10        640: 100%|██████████| 13/13 [01:00<00:00,  4.64s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.412      0.324      0.328      0.125

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/300         0G      1.872      1.856      1.696         10        640: 100%|██████████| 13/13 [01:00<00:00,  4.67s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74      0.209      0.405      0.205     0.0908

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/300         0G      1.884      1.859      1.747          7        640: 100%|██████████| 13/13 [01:00<00:00,  4.63s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.311      0.232      0.176     0.0694

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/300         0G      1.962      1.969      1.828          4        640: 100%|██████████| 13/13 [01:00<00:00,  4.64s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.54s/it]
                   all         69         74      0.206      0.149      0.135     0.0518

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/300         0G      1.856      1.773      1.703          7        640: 100%|██████████| 13/13 [01:00<00:00,  4.63s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.54s/it]
                   all         69         74      0.271      0.149      0.138     0.0467

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/300         0G      1.779      1.568      1.645          8        640: 100%|██████████| 13/13 [01:00<00:00,  4.64s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.56s/it]
                   all         69         74      0.523      0.415      0.369      0.164

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/300         0G      1.819      1.709      1.649          5        640: 100%|██████████| 13/13 [01:00<00:00,  4.66s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.55s/it]
                   all         69         74      0.416      0.298      0.295      0.129

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/300         0G      1.905      1.738      1.716         13        640: 100%|██████████| 13/13 [01:00<00:00,  4.65s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74       0.65      0.376      0.467      0.137

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/300         0G      1.742      1.543      1.627         12        640: 100%|██████████| 13/13 [01:00<00:00,  4.63s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.55s/it]
                   all         69         74     0.0513      0.486     0.0604     0.0206

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/300         0G      1.813      1.518      1.655          8        640: 100%|██████████| 13/13 [01:00<00:00,  4.64s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.56s/it]
                   all         69         74      0.232      0.392      0.218     0.0936

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/300         0G      1.847       1.49      1.714         14        640: 100%|██████████| 13/13 [01:00<00:00,  4.64s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.55s/it]
                   all         69         74      0.299      0.432      0.283      0.125

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/300         0G      1.769      1.476      1.642          9        640: 100%|██████████| 13/13 [01:01<00:00,  4.69s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.425      0.608       0.46      0.219

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/300         0G      1.736      1.419      1.622          8        640: 100%|██████████| 13/13 [01:00<00:00,  4.62s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.592      0.432      0.476      0.201

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/300         0G      1.802      1.422      1.652         11        640: 100%|██████████| 13/13 [01:00<00:00,  4.66s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.67s/it]
                   all         69         74      0.607      0.595      0.631       0.25

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/300         0G      1.782      1.377      1.662         10        640: 100%|██████████| 13/13 [01:04<00:00,  4.98s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.61s/it]
                   all         69         74      0.612      0.622      0.667      0.288

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/300         0G       1.74      1.352      1.629         12        640: 100%|██████████| 13/13 [32:23<00:00, 149.50s/it] 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.59s/it]
                   all         69         74      0.756      0.676      0.725       0.33

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/300         0G      1.682      1.344      1.621          9        640: 100%|██████████| 13/13 [01:00<00:00,  4.67s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74      0.744      0.662      0.744      0.305

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/300         0G      1.673      1.293      1.596          8        640: 100%|██████████| 13/13 [17:06<00:00, 78.95s/it] 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.59s/it]
                   all         69         74      0.695      0.689      0.725      0.323

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/300         0G      1.684      1.247      1.567          7        640: 100%|██████████| 13/13 [10:51<00:00, 50.10s/it] 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.61s/it]
                   all         69         74      0.692      0.676      0.722      0.331

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/300         0G      1.638      1.261      1.554         17        640: 100%|██████████| 13/13 [1:11:56<00:00, 332.05s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [03:23<00:00, 67.79s/it] 
                   all         69         74      0.784       0.77      0.831      0.352

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/300         0G       1.69      1.247      1.581          9        640: 100%|██████████| 13/13 [2:47:57<00:00, 775.18s/it] 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [15:10<00:00, 303.41s/it]
                   all         69         74      0.723       0.77      0.815      0.356

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/300         0G      1.667      1.234      1.545         10        640: 100%|██████████| 13/13 [2:32:20<00:00, 703.09s/it] 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [16:23<00:00, 327.99s/it]
                   all         69         74      0.752       0.73      0.772      0.347

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/300         0G      1.667      1.266      1.541          7        640: 100%|██████████| 13/13 [2:10:47<00:00, 603.69s/it] 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [33:52<00:00, 677.57s/it] 
                   all         69         74      0.675      0.716      0.639       0.29

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/300         0G      1.673      1.175      1.561         10        640: 100%|██████████| 13/13 [1:39:15<00:00, 458.15s/it] 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [14:23<00:00, 287.72s/it]
                   all         69         74      0.711      0.649      0.731      0.342

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/300         0G      1.681      1.238      1.565         12        640: 100%|██████████| 13/13 [01:00<00:00,  4.68s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.59s/it]
                   all         69         74      0.662       0.77      0.712      0.356

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/300         0G      1.616      1.199      1.531          6        640: 100%|██████████| 13/13 [01:00<00:00,  4.67s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.59s/it]
                   all         69         74      0.798      0.784       0.83      0.384

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/300         0G      1.671      1.228      1.571         13        640: 100%|██████████| 13/13 [01:00<00:00,  4.66s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.764      0.743      0.779      0.372

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/300         0G      1.597      1.253      1.507         10        640: 100%|██████████| 13/13 [01:00<00:00,  4.65s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.883       0.73      0.813      0.352

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/300         0G      1.704      1.203       1.55         13        640: 100%|██████████| 13/13 [01:54<00:00,  8.85s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.60s/it]
                   all         69         74      0.754      0.789       0.78      0.341

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/300         0G      1.689      1.184      1.553         11        640: 100%|██████████| 13/13 [01:01<00:00,  4.76s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.76s/it]
                   all         69         74      0.736       0.73      0.768      0.324

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/300         0G      1.657      1.173      1.537         10        640: 100%|██████████| 13/13 [01:01<00:00,  4.74s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.91s/it]
                   all         69         74      0.736      0.678      0.771      0.335

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/300         0G       1.63      1.134      1.549         10        640: 100%|██████████| 13/13 [01:01<00:00,  4.74s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.62s/it]
                   all         69         74      0.798      0.689      0.775       0.34

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/300         0G      1.604      1.201      1.498         11        640: 100%|██████████| 13/13 [01:01<00:00,  4.77s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.67s/it]
                   all         69         74      0.778      0.811      0.821      0.347

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/300         0G      1.677       1.19      1.542         11        640: 100%|██████████| 13/13 [01:01<00:00,  4.76s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.63s/it]
                   all         69         74      0.719      0.851       0.81      0.348

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/300         0G      1.506        1.1      1.463         14        640: 100%|██████████| 13/13 [01:01<00:00,  4.74s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.60s/it]
                   all         69         74      0.833      0.811      0.817      0.343

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/300         0G      1.637      1.191      1.513         12        640: 100%|██████████| 13/13 [01:01<00:00,  4.73s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74      0.806      0.786      0.811      0.359

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/300         0G       1.55      1.083      1.511          9        640: 100%|██████████| 13/13 [01:00<00:00,  4.68s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.851      0.769      0.807      0.347

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/300         0G      1.606       1.18      1.508          7        640: 100%|██████████| 13/13 [01:01<00:00,  4.71s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.59s/it]
                   all         69         74      0.728      0.832      0.813       0.37

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/300         0G       1.52      1.126      1.484          7        640: 100%|██████████| 13/13 [01:00<00:00,  4.67s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.59s/it]
                   all         69         74      0.817      0.905      0.868       0.38

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/300         0G      1.539      1.044      1.509          8        640: 100%|██████████| 13/13 [01:00<00:00,  4.66s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.56s/it]
                   all         69         74      0.799      0.824      0.867      0.384

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/300         0G      1.609      1.067      1.525         10        640: 100%|██████████| 13/13 [01:00<00:00,  4.66s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.798       0.73      0.815       0.39

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/300         0G      1.635      1.119      1.512          7        640: 100%|██████████| 13/13 [01:00<00:00,  4.65s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.56s/it]
                   all         69         74      0.858      0.784      0.863      0.392

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/300         0G      1.624       1.14      1.507         15        640: 100%|██████████| 13/13 [01:03<00:00,  4.87s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.66s/it]
                   all         69         74      0.797      0.847       0.82      0.379

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/300         0G      1.558      1.114      1.507         10        640: 100%|██████████| 13/13 [01:01<00:00,  4.72s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.60s/it]
                   all         69         74      0.859      0.797      0.857      0.406

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/300         0G      1.555      1.071      1.496          4        640: 100%|██████████| 13/13 [01:01<00:00,  4.77s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.62s/it]
                   all         69         74      0.771      0.773      0.826      0.383

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/300         0G      1.523      1.095      1.477          6        640: 100%|██████████| 13/13 [01:01<00:00,  4.70s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.68s/it]
                   all         69         74      0.797      0.797      0.796      0.359

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/300         0G        1.5       1.03      1.422          6        640: 100%|██████████| 13/13 [01:03<00:00,  4.85s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.69s/it]
                   all         69         74      0.833      0.797      0.866      0.382

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/300         0G      1.507      1.011      1.491          6        640: 100%|██████████| 13/13 [01:02<00:00,  4.79s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.75s/it]
                   all         69         74      0.727      0.792      0.816      0.367

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/300         0G       1.52      1.006      1.484          9        640: 100%|██████████| 13/13 [01:01<00:00,  4.75s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.73s/it]
                   all         69         74      0.684       0.77      0.748      0.354

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/300         0G      1.469      1.046      1.438          5        640: 100%|██████████| 13/13 [01:03<00:00,  4.86s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74      0.777      0.824      0.812      0.377

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/300         0G      1.483      1.048      1.448          9        640: 100%|██████████| 13/13 [01:03<00:00,  4.92s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74       0.86      0.797       0.87      0.388

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/300         0G       1.47     0.9923      1.428          8        640: 100%|██████████| 13/13 [01:01<00:00,  4.75s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.68s/it]
                   all         69         74      0.792      0.875      0.869      0.402

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/300         0G      1.473      1.026      1.458          8        640: 100%|██████████| 13/13 [01:03<00:00,  4.86s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.70s/it]
                   all         69         74      0.773      0.876      0.855      0.393

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/300         0G      1.549      1.075      1.507         10        640: 100%|██████████| 13/13 [01:05<00:00,  5.01s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.63s/it]
                   all         69         74      0.773      0.851      0.846      0.391

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/300         0G      1.469       1.01      1.474         10        640: 100%|██████████| 13/13 [01:04<00:00,  4.94s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.60s/it]
                   all         69         74      0.821      0.807      0.846      0.385

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/300         0G      1.502      1.027      1.467         10        640: 100%|██████████| 13/13 [01:05<00:00,  5.02s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.60s/it]
                   all         69         74      0.737      0.758      0.814      0.386

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/300         0G      1.453      1.046      1.428          6        640: 100%|██████████| 13/13 [01:06<00:00,  5.14s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.62s/it]
                   all         69         74      0.841      0.824      0.872      0.369

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/300         0G      1.494      1.011      1.455         10        640: 100%|██████████| 13/13 [01:03<00:00,  4.92s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.59s/it]
                   all         69         74      0.867      0.794      0.856      0.379

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/300         0G       1.54      1.049      1.481         12        640: 100%|██████████| 13/13 [01:04<00:00,  4.93s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.84s/it]
                   all         69         74      0.776      0.797      0.839      0.361

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/300         0G      1.387      1.023      1.388          5        640: 100%|██████████| 13/13 [01:03<00:00,  4.89s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.73s/it]
                   all         69         74      0.744      0.851      0.832      0.358

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/300         0G       1.53     0.9933      1.456         12        640: 100%|██████████| 13/13 [01:01<00:00,  4.76s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.72s/it]
                   all         69         74      0.826      0.834      0.854      0.381

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/300         0G      1.386     0.9656      1.404         11        640: 100%|██████████| 13/13 [01:04<00:00,  4.93s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.63s/it]
                   all         69         74      0.794      0.838       0.85      0.386

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/300         0G      1.521      1.043      1.474          8        640: 100%|██████████| 13/13 [01:07<00:00,  5.18s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.82s/it]
                   all         69         74      0.823      0.865      0.857      0.395

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/300         0G      1.519      1.097      1.434          6        640: 100%|██████████| 13/13 [01:04<00:00,  4.95s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.59s/it]
                   all         69         74      0.839       0.77      0.852      0.384

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/300         0G      1.408      1.034      1.414         10        640: 100%|██████████| 13/13 [01:02<00:00,  4.80s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.74s/it]
                   all         69         74      0.875      0.811      0.857      0.362

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/300         0G      1.442     0.9545      1.451         13        640: 100%|██████████| 13/13 [01:07<00:00,  5.18s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.66s/it]
                   all         69         74      0.849      0.865      0.881      0.387

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/300         0G      1.401     0.9434      1.399          8        640: 100%|██████████| 13/13 [01:06<00:00,  5.13s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.70s/it]
                   all         69         74      0.828      0.779      0.856      0.379

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/300         0G       1.49      1.006      1.494          8        640: 100%|██████████| 13/13 [01:09<00:00,  5.35s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.64s/it]
                   all         69         74      0.818      0.791      0.813      0.367

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/300         0G      1.364      1.006      1.436          6        640: 100%|██████████| 13/13 [01:04<00:00,  4.96s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.72s/it]
                   all         69         74      0.816       0.84      0.844      0.376

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/300         0G      1.408     0.9663      1.406         11        640: 100%|██████████| 13/13 [01:11<00:00,  5.47s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74      0.847      0.824      0.847      0.373

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/300         0G      1.451      1.026      1.422         11        640: 100%|██████████| 13/13 [01:12<00:00,  5.56s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.99s/it]
                   all         69         74        0.8      0.866      0.874      0.369

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/300         0G      1.435     0.9843       1.42          8        640: 100%|██████████| 13/13 [01:10<00:00,  5.44s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.79s/it]
                   all         69         74      0.788      0.804      0.837      0.373

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/300         0G      1.409     0.9971      1.411          7        640: 100%|██████████| 13/13 [01:09<00:00,  5.34s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:09<00:00,  3.11s/it]
                   all         69         74      0.747      0.905       0.87      0.396

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/300         0G      1.461       1.04      1.428         14        640: 100%|██████████| 13/13 [01:22<00:00,  6.38s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.81s/it]
                   all         69         74      0.785      0.811      0.847      0.379

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/300         0G      1.362     0.9075        1.4         10        640: 100%|██████████| 13/13 [01:20<00:00,  6.20s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:09<00:00,  3.00s/it]
                   all         69         74       0.86       0.77       0.83      0.372

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/300         0G      1.488      1.016      1.465         12        640: 100%|██████████| 13/13 [01:05<00:00,  5.06s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.65s/it]
                   all         69         74      0.825      0.838       0.84        0.4

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/300         0G      1.401     0.9588      1.428         15        640: 100%|██████████| 13/13 [01:21<00:00,  6.29s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:11<00:00,  3.88s/it]
                   all         69         74      0.683       0.77      0.757      0.346

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/300         0G      1.421      1.014      1.409          9        640: 100%|██████████| 13/13 [01:11<00:00,  5.51s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.62s/it]
                   all         69         74      0.849      0.865      0.876      0.365

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/300         0G      1.402      0.998       1.38          8        640: 100%|██████████| 13/13 [01:12<00:00,  5.58s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:09<00:00,  3.25s/it]
                   all         69         74      0.791      0.851      0.843      0.376

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/300         0G      1.351     0.9094      1.361         11        640: 100%|██████████| 13/13 [01:18<00:00,  6.03s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.71s/it]
                   all         69         74      0.816      0.851      0.838      0.362

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/300         0G      1.382     0.9602      1.372          5        640: 100%|██████████| 13/13 [01:06<00:00,  5.12s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.61s/it]
                   all         69         74      0.837      0.835      0.878        0.4

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/300         0G      1.416     0.9394      1.407          9        640: 100%|██████████| 13/13 [01:03<00:00,  4.89s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:09<00:00,  3.18s/it]
                   all         69         74      0.795      0.784      0.829       0.37

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/300         0G      1.406     0.8985      1.404          8        640: 100%|██████████| 13/13 [01:08<00:00,  5.28s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.86s/it]
                   all         69         74      0.825      0.764      0.837      0.365

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/300         0G      1.411     0.9305      1.404          7        640: 100%|██████████| 13/13 [01:04<00:00,  4.98s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.84s/it]
                   all         69         74      0.797      0.851      0.871      0.413

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/300         0G      1.388      1.002      1.402          6        640: 100%|██████████| 13/13 [01:05<00:00,  5.08s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.70s/it]
                   all         69         74      0.832      0.784      0.859      0.411

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/300         0G      1.392     0.9439      1.405         12        640: 100%|██████████| 13/13 [01:03<00:00,  4.86s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.825      0.831       0.85      0.376

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/300         0G      1.465      1.011       1.44         11        640: 100%|██████████| 13/13 [01:02<00:00,  4.84s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.68s/it]
                   all         69         74      0.795      0.824      0.841      0.386

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/300         0G      1.431     0.9505      1.391         12        640: 100%|██████████| 13/13 [01:04<00:00,  4.98s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.63s/it]
                   all         69         74      0.833      0.851      0.868      0.397

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/300         0G      1.425     0.9571      1.399          8        640: 100%|██████████| 13/13 [01:03<00:00,  4.88s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.90s/it]
                   all         69         74       0.82      0.851      0.869      0.402

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/300         0G      1.451     0.9615      1.404          8        640: 100%|██████████| 13/13 [01:04<00:00,  4.94s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.84s/it]
                   all         69         74      0.875      0.784      0.831      0.377

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/300         0G      1.314     0.9123      1.339         11        640: 100%|██████████| 13/13 [01:07<00:00,  5.23s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:09<00:00,  3.08s/it]
                   all         69         74      0.892      0.865      0.881      0.395

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    101/300         0G      1.364     0.9093      1.382          8        640: 100%|██████████| 13/13 [01:08<00:00,  5.29s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.62s/it]
                   all         69         74      0.816        0.9      0.871      0.401

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    102/300         0G      1.406     0.9434      1.367          9        640: 100%|██████████| 13/13 [01:05<00:00,  5.05s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.66s/it]
                   all         69         74      0.859      0.822      0.876      0.399

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    103/300         0G      1.443     0.9677      1.417          4        640: 100%|██████████| 13/13 [01:04<00:00,  4.97s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:09<00:00,  3.14s/it]
                   all         69         74      0.809      0.878      0.885      0.411

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    104/300         0G      1.441     0.9604      1.441         10        640: 100%|██████████| 13/13 [01:07<00:00,  5.23s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:09<00:00,  3.05s/it]
                   all         69         74      0.829      0.878      0.869      0.397

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    105/300         0G      1.371      0.912      1.381         13        640: 100%|██████████| 13/13 [01:07<00:00,  5.16s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.70s/it]
                   all         69         74       0.82      0.799      0.853      0.397

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    106/300         0G      1.352     0.9217      1.362          9        640: 100%|██████████| 13/13 [01:03<00:00,  4.90s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.60s/it]
                   all         69         74       0.86      0.811      0.873      0.396

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    107/300         0G      1.316     0.8643      1.329         14        640: 100%|██████████| 13/13 [01:03<00:00,  4.87s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.65s/it]
                   all         69         74      0.851      0.878      0.869      0.403

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    108/300         0G      1.378     0.9376      1.374          7        640: 100%|██████████| 13/13 [01:02<00:00,  4.80s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.79s/it]
                   all         69         74      0.816      0.837      0.865      0.399

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    109/300         0G      1.381     0.9138      1.369          6        640: 100%|██████████| 13/13 [01:06<00:00,  5.11s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.70s/it]
                   all         69         74      0.831      0.795      0.879      0.379

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    110/300         0G      1.336     0.9064      1.348          6        640: 100%|██████████| 13/13 [01:04<00:00,  4.99s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.79s/it]
                   all         69         74      0.837      0.892      0.887      0.393

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    111/300         0G      1.335     0.9011      1.363         11        640: 100%|██████████| 13/13 [01:03<00:00,  4.92s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.70s/it]
                   all         69         74      0.875      0.892       0.89      0.404

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    112/300         0G      1.318     0.8956      1.346          8        640: 100%|██████████| 13/13 [01:06<00:00,  5.14s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.70s/it]
                   all         69         74      0.831      0.861      0.863      0.399

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    113/300         0G      1.256     0.8776      1.335          7        640: 100%|██████████| 13/13 [01:03<00:00,  4.91s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.68s/it]
                   all         69         74      0.798      0.838      0.831      0.385

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    114/300         0G      1.288     0.8581      1.338          6        640: 100%|██████████| 13/13 [01:04<00:00,  4.95s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.79s/it]
                   all         69         74        0.8      0.867      0.855      0.395

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    115/300         0G      1.286      0.898      1.318          8        640: 100%|██████████| 13/13 [01:02<00:00,  4.85s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.64s/it]
                   all         69         74      0.816      0.841      0.889      0.412

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    116/300         0G      1.354     0.8794      1.352         10        640: 100%|██████████| 13/13 [01:03<00:00,  4.91s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.89s/it]
                   all         69         74      0.808      0.838       0.89      0.409

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    117/300         0G      1.229     0.8328        1.3          8        640: 100%|██████████| 13/13 [01:04<00:00,  4.95s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.67s/it]
                   all         69         74      0.841      0.851      0.885      0.413

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    118/300         0G      1.335     0.8574      1.336          8        640: 100%|██████████| 13/13 [01:06<00:00,  5.12s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:09<00:00,  3.06s/it]
                   all         69         74      0.869      0.809      0.884      0.397

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    119/300         0G      1.308      0.878      1.326          9        640: 100%|██████████| 13/13 [01:06<00:00,  5.10s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:09<00:00,  3.08s/it]
                   all         69         74       0.81      0.805      0.847      0.397

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    120/300         0G      1.289     0.8973      1.336         11        640: 100%|██████████| 13/13 [01:03<00:00,  4.88s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.67s/it]
                   all         69         74       0.81      0.865      0.879      0.425

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    121/300         0G      1.347     0.8748      1.371          8        640: 100%|██████████| 13/13 [01:02<00:00,  4.84s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.66s/it]
                   all         69         74      0.841      0.858      0.874      0.403

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    122/300         0G      1.412     0.9173      1.417         10        640: 100%|██████████| 13/13 [01:03<00:00,  4.90s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.70s/it]
                   all         69         74      0.848      0.831      0.875      0.397

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    123/300         0G      1.346     0.8699      1.368         11        640: 100%|██████████| 13/13 [01:07<00:00,  5.19s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.88s/it]
                   all         69         74      0.809      0.859      0.881        0.4

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    124/300         0G      1.273     0.8444      1.328          4        640: 100%|██████████| 13/13 [01:05<00:00,  5.04s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.65s/it]
                   all         69         74       0.86      0.865      0.891      0.396

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    125/300         0G      1.386     0.9391      1.384          7        640: 100%|██████████| 13/13 [01:05<00:00,  5.08s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.81s/it]
                   all         69         74      0.811      0.867      0.882       0.41

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    126/300         0G      1.326     0.9138      1.364          5        640: 100%|██████████| 13/13 [01:04<00:00,  4.98s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:09<00:00,  3.02s/it]
                   all         69         74      0.813      0.762      0.848      0.373

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    127/300         0G      1.329     0.9097      1.381          8        640: 100%|██████████| 13/13 [01:02<00:00,  4.83s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.66s/it]
                   all         69         74      0.851      0.784      0.848      0.379

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    128/300         0G      1.313     0.8715      1.332         10        640: 100%|██████████| 13/13 [01:07<00:00,  5.21s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.90s/it]
                   all         69         74      0.849      0.784      0.881      0.394

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    129/300         0G      1.244      0.811      1.309          8        640: 100%|██████████| 13/13 [01:05<00:00,  5.03s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.91s/it]
                   all         69         74      0.848      0.824      0.848      0.397

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    130/300         0G      1.247     0.8039      1.289          8        640: 100%|██████████| 13/13 [01:10<00:00,  5.40s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.93s/it]
                   all         69         74      0.844      0.851      0.859      0.384

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    131/300         0G      1.269     0.8186      1.301         10        640: 100%|██████████| 13/13 [01:08<00:00,  5.24s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.97s/it]
                   all         69         74      0.812      0.811      0.843      0.394

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    132/300         0G      1.223     0.8374      1.298         14        640: 100%|██████████| 13/13 [01:08<00:00,  5.28s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.91s/it]
                   all         69         74      0.738      0.838      0.817      0.392

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    133/300         0G      1.349     0.8829      1.335          7        640: 100%|██████████| 13/13 [01:06<00:00,  5.09s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.72s/it]
                   all         69         74      0.819      0.797      0.852      0.388

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    134/300         0G      1.277     0.8607       1.29          5        640: 100%|██████████| 13/13 [01:06<00:00,  5.14s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.79s/it]
                   all         69         74      0.848      0.838      0.878      0.392

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    135/300         0G      1.234     0.8284      1.303          7        640: 100%|██████████| 13/13 [01:04<00:00,  4.97s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:09<00:00,  3.02s/it]
                   all         69         74      0.845      0.865      0.874        0.4

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    136/300         0G      1.244     0.8836      1.311          9        640: 100%|██████████| 13/13 [01:03<00:00,  4.89s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:09<00:00,  3.02s/it]
                   all         69         74      0.855      0.851      0.886      0.416

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    137/300         0G      1.278     0.8418      1.335          5        640: 100%|██████████| 13/13 [01:05<00:00,  5.08s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.87s/it]
                   all         69         74        0.8      0.865      0.876      0.394

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    138/300         0G      1.281     0.8494      1.321         14        640: 100%|██████████| 13/13 [01:05<00:00,  5.00s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.65s/it]
                   all         69         74      0.842      0.851      0.879      0.379

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    139/300         0G      1.212     0.8285      1.291         11        640: 100%|██████████| 13/13 [01:04<00:00,  4.98s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.79s/it]
                   all         69         74      0.819      0.859      0.869      0.394

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    140/300         0G      1.267     0.8622      1.316          6        640: 100%|██████████| 13/13 [01:04<00:00,  4.95s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.81s/it]
                   all         69         74      0.767      0.865      0.865        0.4

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    141/300         0G      1.277     0.8833      1.309         10        640: 100%|██████████| 13/13 [01:01<00:00,  4.73s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74      0.761      0.819      0.859      0.377

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    142/300         0G       1.26     0.8532      1.308          9        640: 100%|██████████| 13/13 [01:00<00:00,  4.67s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.797       0.77      0.855       0.39

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    143/300         0G      1.258     0.8191      1.321         11        640: 100%|██████████| 13/13 [01:00<00:00,  4.66s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74      0.797      0.848      0.851      0.395

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    144/300         0G      1.265     0.8713      1.312         10        640: 100%|██████████| 13/13 [01:00<00:00,  4.66s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.55s/it]
                   all         69         74      0.742      0.779      0.821      0.383

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    145/300         0G      1.267      0.883      1.314         12        640: 100%|██████████| 13/13 [01:02<00:00,  4.82s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.64s/it]
                   all         69         74      0.759      0.811      0.844      0.383

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    146/300         0G      1.229     0.8662       1.29         10        640: 100%|██████████| 13/13 [01:01<00:00,  4.75s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.821      0.838      0.867      0.387

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    147/300         0G       1.22     0.8183      1.288         12        640: 100%|██████████| 13/13 [01:00<00:00,  4.69s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.60s/it]
                   all         69         74      0.798      0.838      0.862      0.406

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    148/300         0G      1.227     0.8316      1.276          7        640: 100%|██████████| 13/13 [01:01<00:00,  4.76s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.60s/it]
                   all         69         74      0.822      0.876       0.88      0.414

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    149/300         0G      1.323     0.8743      1.303          9        640: 100%|██████████| 13/13 [01:01<00:00,  4.75s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74      0.907      0.795      0.876      0.393

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    150/300         0G      1.256     0.8543      1.352          6        640: 100%|██████████| 13/13 [01:01<00:00,  4.72s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74      0.866      0.785      0.874      0.401

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    151/300         0G      1.189      0.844      1.261          4        640: 100%|██████████| 13/13 [01:02<00:00,  4.78s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.65s/it]
                   all         69         74      0.819      0.838      0.865        0.4

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    152/300         0G      1.267     0.8686      1.311         13        640: 100%|██████████| 13/13 [01:01<00:00,  4.76s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.65s/it]
                   all         69         74      0.831      0.838      0.869      0.414

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    153/300         0G      1.174     0.8256      1.263         10        640: 100%|██████████| 13/13 [01:01<00:00,  4.76s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.66s/it]
                   all         69         74      0.878      0.865      0.877      0.418

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    154/300         0G      1.282     0.8575      1.299          6        640: 100%|██████████| 13/13 [01:01<00:00,  4.70s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74      0.862      0.865      0.882      0.403

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    155/300         0G      1.198     0.8704      1.277         11        640: 100%|██████████| 13/13 [01:01<00:00,  4.70s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.65s/it]
                   all         69         74      0.843        0.8      0.855      0.388

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    156/300         0G      1.251     0.8531       1.29          8        640: 100%|██████████| 13/13 [01:02<00:00,  4.79s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.63s/it]
                   all         69         74      0.782       0.77      0.834       0.39

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    157/300         0G      1.258     0.8388       1.31          8        640: 100%|██████████| 13/13 [01:00<00:00,  4.68s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.66s/it]
                   all         69         74      0.804      0.851      0.861      0.409

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    158/300         0G      1.183     0.8479      1.282          8        640: 100%|██████████| 13/13 [1:10:21<00:00, 324.74s/it] 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:36<00:00, 12.14s/it]
                   all         69         74        0.8      0.809      0.851      0.408

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    159/300         0G      1.214     0.8791      1.295          9        640: 100%|██████████| 13/13 [11:23<00:00, 52.59s/it] 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [02:58<00:00, 59.43s/it] 
                   all         69         74      0.761      0.811      0.807      0.388

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    160/300         0G      1.134     0.8164      1.249         14        640: 100%|██████████| 13/13 [1:30:49<00:00, 419.17s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.56s/it]
                   all         69         74      0.744      0.864      0.792      0.377

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    161/300         0G      1.182     0.8281      1.251         16        640: 100%|██████████| 13/13 [05:37<00:00, 25.94s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:19<00:00,  6.39s/it]
                   all         69         74      0.766      0.851      0.844       0.39

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    162/300         0G      1.208      0.816      1.293          9        640: 100%|██████████| 13/13 [03:02<00:00, 14.07s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.58s/it]
                   all         69         74      0.831      0.865      0.877      0.406

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    163/300         0G      1.176     0.7817      1.262         12        640: 100%|██████████| 13/13 [01:01<00:00,  4.76s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.77s/it]
                   all         69         74      0.786      0.865      0.871      0.387

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    164/300         0G      1.219     0.8223      1.285          8        640: 100%|██████████| 13/13 [01:03<00:00,  4.89s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:08<00:00,  2.69s/it]
                   all         69         74      0.848      0.797      0.853      0.366

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    165/300         0G      1.169     0.8195      1.261         10        640: 100%|██████████| 13/13 [01:00<00:00,  4.68s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.60s/it]
                   all         69         74      0.825      0.797      0.859      0.382

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    166/300         0G      1.198     0.8512      1.281          7        640: 100%|██████████| 13/13 [01:00<00:00,  4.65s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.882      0.804      0.869      0.382

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    167/300         0G      1.174     0.8178      1.274         40        640:      167/300         0G      1.175     0.8197      1.273         32        640:      167/300         0G      1.175     0.8197      1.273         32        640:      167/300         0G      1.166      0.809      1.279          6        640:      167/300         0G      1.166      0.809      1.279          6        640: 1    167/300         0G      1.166      0.809      1.279          6        640: 100%|██████████| 13/13 [01:01<00:00,  4.70s/it]
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         69         74      0.846      0.838      0.876      0.378

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    168/300         0G      1.247     0.8958      1.293         12        640: 1
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.59s/it]
                   all         69         74      0.804      0.838      0.856      0.383

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    169/300         0G      1.113     0.8187      1.266          5        640: 100%|██████████| 13/13 [01:00<00:00,  4.65s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.57s/it]
                   all         69         74      0.808      0.853      0.864      0.376

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    170/300         0G      1.156     0.8336      1.244          9        640: 100%|██████████| 13/13 [01:03<00:00,  4.87s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.64s/it]
                   all         69         74      0.831      0.862      0.872      0.387
Stopping training early as no improvement observed in last 50 epochs. Best results observed at epoch 120, best model saved as best.pt.
To update EarlyStopping(patience=50) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

170 epochs completed in 19.281 hours.
Optimizer stripped from /Users/mainfolder/Yolo_Label/runs/detect/train11/weights/last.pt, 6.3MB
Optimizer stripped from /Users/mainfolder/Yolo_Label/runs/detect/train11/weights/best.pt, 6.3MB

Validating /Users/mainfolder/Yolo_Label/runs/detect/train11/weights/best.pt...
Ultralytics YOLOv8.0.223 🚀 Python-3.9.6 torch-2.1.1 CPU (Apple M2)
Model summary (fused): 168 layers, 3005843 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:07<00:00,  2.54s/it]
                   all         69         74       0.81      0.865      0.879      0.425
Speed: 0.7ms preprocess, 102.9ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /Users/mainfolder/Yolo_Label/runs/detect/train11
ultralytics.utils.metrics.DetMetrics object with attributes:

ap_class_index: array([0])
box: ultralytics.utils.metrics.Metric object
confusion_matrix: <ultralytics.utils.metrics.ConfusionMatrix object at 0x291d50b20>
curves: ['Precision-Recall(B)', 'F1-Confidence(B)', 'Precision-Confidence(B)', 'Recall-Confidence(B)']
curves_results: [[array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
          0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
          0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
          0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
          0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
           0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
           0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
           0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
           0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
           0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
           0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
           0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
           0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
           0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
           0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
           0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
           0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
           0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
           0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
           0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
           0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
            0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
           0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
           0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
           0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
            0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
           0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
           0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
           0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
            0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
           0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
           0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
           0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
           0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
           0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
           0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
           0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
           0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
           0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
           0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
           0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
           0.98498,     0.98599,     0.98699,     0.98799,     0.98899,     0.98999,     0.99099,     0.99199,     0.99299,     0.99399,     0.99499,       0.996,       0.997,       0.998,       0.999,           1]), array([[          1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,
             0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,      0.9697,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,
            0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,
            0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,
            0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94872,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,
            0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,
            0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,
            0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,
            0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,
            0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,
            0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,
            0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,
            0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.94444,     0.93103,
            0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,
            0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.93103,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,
            0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.91803,     0.84286,     0.84286,     0.84286,
            0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,
            0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.84286,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,
            0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.83562,     0.82667,     0.82667,     0.82667,     0.82667,
            0.82667,     0.82667,     0.82667,     0.82667,     0.82667,     0.82667,     0.82667,     0.82667,     0.82667,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,
             0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,      0.8125,
             0.8125,      0.8125,      0.8125,      0.8125,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,
            0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.69792,     0.59649,     0.59649,     0.59649,     0.59649,     0.59649,     0.59649,     0.59649,     0.59649,     0.59649,     0.59649,     0.59649,     0.59649,     0.59649,     0.59649,     0.48252,
            0.48252,     0.48252,     0.48252,     0.48252,     0.48252,     0.48252,     0.48252,     0.48252,     0.48252,     0.48252,     0.48252,     0.48252,     0.46053,     0.46053,     0.46053,     0.46053,     0.46053,     0.46053,     0.46053,     0.46053,     0.46053,     0.46053,     0.46053,
            0.46053,     0.46053,     0.14431,     0.14431,     0.14431,     0.14431,     0.14431,     0.14431,     0.14431,     0.14431,     0.14431,     0.14431,     0.14431,     0.14431,     0.14431,     0.14431,    0.052527,    0.051214,    0.049901,    0.048587,    0.047274,    0.045961,    0.044648,
           0.043335,    0.042022,    0.040708,    0.039395,    0.038082,    0.036769,    0.035456,    0.034143,    0.032829,    0.031516,    0.030203,     0.02889,    0.027577,    0.026263,     0.02495,    0.023637,    0.022324,    0.021011,    0.019698,    0.018384,    0.017071,    0.015758,    0.014445,
           0.013132,    0.011819,    0.010505,   0.0091922,    0.007879,   0.0065659,   0.0052527,   0.0039395,   0.0026263,   0.0013132,           0]]), 'Recall', 'Precision'], [array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
          0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
          0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
          0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
          0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
           0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
           0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
           0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
           0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
           0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
           0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
           0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
           0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
           0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
           0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
           0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
           0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
           0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
           0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
           0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
           0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
            0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
           0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
           0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
           0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
            0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
           0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
           0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
           0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
            0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
           0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
           0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
           0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
           0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
           0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
           0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
           0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
           0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
           0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
           0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
           0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
           0.98498,     0.98599,     0.98699,     0.98799,     0.98899,     0.98999,     0.99099,     0.99199,     0.99299,     0.99399,     0.99499,       0.996,       0.997,       0.998,       0.999,           1]), array([[    0.10078,     0.10088,     0.17034,     0.21608,     0.24778,     0.26786,     0.29205,     0.31128,     0.31799,     0.33223,     0.34401,      0.3601,     0.37209,     0.37751,     0.38363,     0.39236,     0.40346,     0.41502,     0.42152,     0.43239,     0.43527,     0.44492,     0.45437,
            0.45829,     0.46089,     0.47067,      0.4725,     0.48087,     0.48524,     0.49468,     0.50413,     0.50521,     0.50719,     0.51365,     0.51948,     0.52458,     0.52981,     0.53915,     0.54051,     0.55078,     0.55278,     0.56076,     0.56194,     0.56835,     0.57253,     0.58585,
            0.58889,     0.59494,       0.597,     0.59929,     0.60912,     0.61009,     0.61107,     0.61876,     0.61623,     0.61369,     0.61493,     0.61977,     0.62883,     0.63483,     0.63363,     0.63243,     0.63123,     0.63002,     0.63009,     0.63078,     0.63146,     0.63214,      0.6361,
            0.63759,     0.64133,     0.64475,     0.64496,     0.64517,     0.64538,     0.64559,      0.6458,     0.64601,     0.64622,     0.64642,     0.64663,     0.64684,     0.64705,     0.64726,     0.64747,     0.64831,     0.65388,     0.65428,     0.65469,     0.65509,      0.6555,      0.6559,
            0.65631,     0.65671,     0.66033,     0.66084,     0.66134,     0.66185,     0.66235,     0.66286,     0.66336,     0.66764,     0.66873,     0.66981,     0.68028,     0.68144,      0.6826,     0.68708,     0.68779,     0.68851,     0.68922,     0.68993,     0.69115,     0.69312,     0.69468,
            0.69598,     0.69728,     0.69926,     0.70504,     0.70771,     0.71277,     0.71373,     0.71468,     0.71563,     0.71666,      0.7177,     0.71874,     0.72031,     0.72331,      0.7228,     0.72229,     0.72179,     0.72128,     0.72077,     0.72026,     0.71975,     0.71924,     0.71872,
            0.71821,      0.7177,     0.71719,     0.71668,     0.71734,     0.71829,     0.71925,      0.7202,     0.72066,     0.72097,     0.72127,     0.72158,     0.72189,     0.72219,      0.7225,      0.7228,     0.72311,     0.72341,     0.72372,     0.72402,     0.72827,     0.72884,     0.72942,
            0.72999,     0.73056,     0.73113,      0.7317,     0.73695,     0.74219,     0.74474,     0.74717,     0.74973,     0.75245,     0.75756,     0.75813,      0.7587,     0.75927,     0.75984,     0.76041,     0.76098,     0.76141,     0.76156,     0.76172,     0.76187,     0.76202,     0.76217,
            0.76233,     0.76248,     0.76263,     0.76278,     0.76294,     0.76309,     0.76324,     0.76339,     0.76354,     0.76369,     0.76385,       0.764,     0.76415,      0.7643,     0.76445,     0.76461,     0.76476,     0.76491,     0.76506,     0.76521,     0.76536,     0.76551,     0.76567,
            0.77022,     0.77038,     0.77053,     0.77069,     0.77085,       0.771,     0.77116,     0.77131,     0.77147,     0.77163,     0.77178,     0.77194,     0.77209,     0.77225,      0.7724,     0.77256,     0.77272,     0.77287,     0.77303,     0.77318,     0.77334,     0.77349,     0.77365,
             0.7738,     0.77396,     0.77411,     0.77427,     0.77442,     0.77991,     0.78383,     0.78417,      0.7845,     0.78483,     0.78516,     0.78549,     0.78582,     0.78615,     0.78648,     0.78681,     0.78714,     0.78747,      0.7878,     0.78813,     0.78723,     0.78577,      0.7843,
            0.78284,     0.78137,     0.77933,     0.77714,     0.77495,     0.77855,     0.77879,     0.77902,     0.77925,     0.77948,     0.77971,     0.77994,     0.78017,      0.7804,     0.78063,     0.78086,     0.78109,     0.78132,     0.78155,     0.78178,     0.78201,     0.78224,     0.78247,
             0.7827,     0.78293,     0.78807,     0.78971,     0.79134,     0.79299,     0.79476,     0.79652,     0.79779,     0.79836,     0.79893,      0.7995,     0.80007,     0.80064,     0.80121,     0.80178,     0.80235,     0.80785,     0.80884,     0.80983,     0.81082,      0.8118,     0.81263,
            0.81309,     0.81355,     0.81401,     0.81447,     0.81492,     0.81538,     0.81583,     0.81629,     0.81674,      0.8172,     0.82283,     0.82329,     0.82375,     0.82421,     0.82468,     0.82514,      0.8256,     0.82605,     0.82651,     0.82697,     0.82743,     0.82789,     0.83378,
            0.83441,     0.83504,     0.83568,     0.83631,     0.83694,     0.83757,      0.8382,     0.83896,     0.84032,     0.84168,     0.84303,     0.84404,     0.84333,     0.84263,     0.84192,     0.84121,      0.8405,     0.83979,     0.83908,     0.83837,     0.83766,     0.83695,     0.82788,
            0.82581,     0.82373,     0.82164,     0.82387,     0.82681,     0.82764,     0.82846,     0.82928,      0.8301,     0.83091,     0.83173,     0.83209,     0.83179,     0.83148,     0.83118,     0.83087,     0.83057,     0.83026,     0.82996,     0.82965,     0.82935,     0.82904,     0.82874,
            0.82843,     0.82813,     0.82782,     0.82751,     0.82721,      0.8269,      0.8266,     0.82629,     0.82598,     0.82568,     0.82537,     0.82506,     0.82476,     0.82445,     0.82902,     0.82746,     0.82591,     0.82435,     0.82279,     0.81444,      0.8159,     0.81736,     0.81881,
            0.81017,     0.80833,     0.80649,     0.80465,     0.80037,     0.79466,     0.79501,     0.79535,      0.7957,     0.79604,     0.79639,     0.79673,     0.79707,     0.79742,     0.79776,      0.7981,     0.79844,     0.79878,     0.79913,     0.79947,     0.79981,      0.8061,     0.80688,
            0.80765,     0.80843,     0.80921,     0.80998,     0.81075,     0.81152,     0.81777,      0.8181,     0.81843,     0.81876,     0.81909,     0.81941,     0.81974,     0.82007,      0.8204,     0.82073,     0.82105,     0.82138,     0.82171,     0.82203,     0.82236,     0.82268,     0.82301,
            0.82333,     0.82401,     0.82521,      0.8264,     0.82759,     0.82878,     0.81973,     0.81561,     0.81207,     0.81233,      0.8126,     0.81287,     0.81313,      0.8134,     0.81366,     0.81393,     0.81419,     0.81446,     0.81472,     0.81498,     0.81525,     0.81551,     0.81578,
            0.81604,      0.8163,     0.81656,     0.81683,     0.81709,     0.81735,     0.81761,     0.81788,     0.81814,       0.815,     0.81113,     0.80672,     0.80173,      0.7993,     0.79824,     0.79717,     0.79611,     0.79504,     0.79397,     0.79289,     0.79182,     0.79074,      0.7925,
            0.79437,     0.79624,     0.79468,     0.79132,     0.78793,     0.78682,     0.78612,     0.78543,     0.78474,     0.78404,     0.78335,     0.78265,     0.78195,     0.78125,     0.78055,     0.77985,     0.77915,     0.77845,     0.77765,     0.77434,       0.771,     0.76787,     0.76665,
            0.76543,      0.7642,     0.76297,     0.76174,      0.7605,     0.75927,     0.75804,     0.75739,     0.75674,     0.75608,     0.75543,     0.75477,     0.75411,     0.75346,      0.7528,     0.75214,     0.75148,     0.75082,     0.75016,      0.7495,     0.74884,     0.74818,     0.73693,
            0.73579,     0.73465,      0.7335,     0.73236,     0.73121,     0.73006,     0.72891,     0.72776,     0.71628,      0.7156,     0.71492,     0.71425,     0.71357,     0.71289,     0.71221,     0.71153,     0.71085,     0.71017,     0.70948,      0.7088,     0.70812,     0.70743,     0.70675,
            0.70606,     0.69394,     0.69261,     0.69128,     0.68995,     0.68861,     0.68727,     0.68593,     0.68459,     0.68322,     0.68182,     0.68042,     0.67901,     0.67761,     0.67619,     0.67478,     0.67336,     0.66035,     0.65877,     0.65719,     0.65561,     0.65402,     0.65243,
            0.65084,     0.64924,     0.64978,     0.65049,     0.65119,     0.65189,     0.65259,     0.65329,     0.65398,     0.65467,     0.65069,     0.64477,     0.62744,      0.6227,     0.61742,     0.60435,     0.59825,     0.59264,     0.59317,     0.59371,     0.59424,     0.59477,      0.5953,
            0.59582,     0.59635,     0.59687,     0.59739,     0.59791,     0.59501,      0.5895,     0.58475,     0.58382,     0.58289,     0.58196,     0.58103,     0.58009,     0.57916,     0.57822,     0.57729,     0.57635,     0.57541,     0.57447,     0.57353,     0.57259,     0.57164,     0.57606,
            0.57494,     0.57382,     0.57269,     0.57156,     0.57044,     0.56931,     0.56817,     0.56704,     0.56591,     0.56477,     0.56363,     0.54988,     0.51962,     0.51922,     0.51882,     0.51842,     0.51802,     0.51762,     0.51723,     0.51683,     0.51643,     0.51603,     0.51563,
            0.51523,     0.51483,     0.51443,     0.51403,     0.51362,     0.51322,     0.51282,     0.51242,     0.51202,     0.51162,     0.51121,     0.51081,     0.51041,     0.51001,      0.5096,      0.5092,      0.5088,     0.50839,     0.50799,     0.50758,     0.50718,     0.50677,     0.50637,
            0.50596,     0.50556,     0.50515,     0.48735,     0.48406,     0.48075,     0.47743,     0.47417,     0.47276,     0.47134,     0.46991,     0.46849,     0.46706,     0.46563,     0.46419,     0.46276,     0.46132,     0.45988,     0.45843,     0.45597,     0.45342,     0.45087,      0.4483,
            0.44573,     0.44315,      0.4415,     0.44049,     0.43947,     0.43845,     0.43743,     0.43641,     0.43539,     0.43437,     0.43334,     0.43232,     0.43129,     0.43026,     0.42923,      0.4282,     0.42717,     0.42614,     0.42431,     0.42138,     0.41843,     0.41548,     0.41251,
            0.40954,     0.39088,     0.38987,     0.38887,     0.38786,     0.38684,     0.38583,     0.38482,      0.3838,     0.38279,     0.38177,     0.38075,     0.37973,     0.37871,     0.37769,     0.37666,     0.37564,     0.37461,     0.37329,     0.36505,     0.35674,     0.35031,     0.34416,
            0.33796,     0.31744,     0.31657,      0.3157,     0.31483,     0.31395,     0.31308,     0.31221,     0.31133,     0.31045,     0.30958,      0.3087,     0.30782,     0.30694,     0.30606,     0.30518,      0.3043,     0.30341,     0.30253,     0.30165,     0.30076,     0.29987,     0.29898,
            0.29811,     0.29723,     0.29635,     0.29547,      0.2946,     0.29372,     0.29283,     0.29195,     0.29107,     0.29019,      0.2893,     0.28842,     0.28753,     0.28664,     0.28575,     0.28486,     0.28397,     0.28308,     0.28219,      0.2813,      0.2804,     0.27951,     0.25764,
            0.25532,     0.25298,     0.25065,      0.2483,     0.24595,      0.2436,     0.24123,     0.23886,      0.2376,     0.23687,     0.23614,     0.23541,     0.23468,     0.23395,     0.23322,     0.23248,     0.23175,     0.23102,     0.23028,     0.22955,     0.22881,     0.22807,     0.22734,
             0.2266,     0.22586,     0.22512,     0.22438,     0.22364,      0.2229,     0.22216,     0.22142,     0.22067,     0.21993,     0.21918,     0.21844,     0.21769,     0.21695,     0.20615,     0.19505,     0.19428,      0.1935,     0.19273,     0.19195,     0.19118,      0.1904,     0.18962,
            0.18884,     0.18806,     0.18728,      0.1865,     0.18572,     0.18494,     0.18416,     0.18338,     0.18259,     0.18181,     0.18103,     0.18024,     0.17945,     0.17867,     0.17788,     0.17709,      0.1763,     0.17551,     0.17472,     0.17393,     0.17314,      0.1717,     0.16987,
            0.16803,     0.16619,     0.16434,      0.1625,     0.16064,     0.15879,     0.15693,     0.15506,     0.15319,     0.15132,     0.14608,     0.13261,     0.12375,     0.11866,     0.11354,     0.10839,     0.10321,     0.10187,     0.10108,     0.10028,    0.099487,    0.098692,    0.097895,
           0.097098,    0.096301,    0.095503,    0.094704,    0.093904,    0.093104,    0.092303,    0.091501,    0.090699,    0.089896,    0.089092,    0.088288,    0.087483,    0.086677,    0.085871,    0.085064,    0.084256,    0.083448,    0.082639,    0.081829,    0.081019,    0.080207,    0.079396,
           0.078583,    0.051226,    0.043677,    0.036069,    0.028402,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,
                  0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,
                  0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,
                  0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,
                  0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0]]), 'Confidence', 'F1'], [array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
          0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
          0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
          0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
          0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
           0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
           0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
           0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
           0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
           0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
           0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
           0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
           0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
           0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
           0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
           0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
           0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
           0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
           0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
           0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
           0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
            0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
           0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
           0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
           0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
            0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
           0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
           0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
           0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
            0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
           0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
           0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
           0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
           0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
           0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
           0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
           0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
           0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
           0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
           0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
           0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
           0.98498,     0.98599,     0.98699,     0.98799,     0.98899,     0.98999,     0.99099,     0.99199,     0.99299,     0.99399,     0.99499,       0.996,       0.997,       0.998,       0.999,           1]), array([[   0.053184,    0.053236,    0.093469,     0.12175,     0.14226,     0.15602,     0.17268,     0.18629,     0.19112,      0.2015,     0.21023,     0.22237,      0.2316,     0.23581,      0.2406,     0.24751,     0.25641,     0.26582,     0.27118,     0.28025,     0.28267,     0.29087,     0.29899,
             0.3024,     0.30467,     0.31327,     0.31489,     0.32237,     0.32631,     0.33491,     0.34363,     0.34464,     0.34648,     0.35254,     0.35805,     0.36292,     0.36794,     0.37702,     0.37835,     0.38849,     0.39048,      0.3985,     0.39968,      0.4062,     0.41049,     0.42432,
            0.42752,     0.43392,     0.43612,     0.43857,     0.44918,     0.45024,      0.4513,     0.46011,     0.45864,     0.45716,     0.45873,     0.46414,     0.47438,     0.48187,     0.48118,     0.48049,     0.47979,      0.4791,     0.47941,      0.4802,     0.48099,     0.48179,      0.4864,
            0.48814,     0.49254,     0.49659,     0.49684,     0.49709,     0.49734,     0.49758,     0.49783,     0.49808,     0.49833,     0.49858,     0.49883,     0.49907,     0.49932,     0.49957,     0.49982,     0.50083,      0.5075,     0.50799,     0.50848,     0.50897,     0.50946,     0.50994,
            0.51043,     0.51092,     0.51532,     0.51594,     0.51655,     0.51717,     0.51779,      0.5184,     0.51902,     0.52428,     0.52562,     0.52696,     0.54004,      0.5415,     0.54297,     0.54866,     0.54957,     0.55048,     0.55139,      0.5523,     0.55386,      0.5564,     0.55841,
             0.5601,     0.56179,     0.56436,     0.57193,     0.57544,     0.58217,     0.58345,     0.58472,     0.58599,     0.58737,     0.58877,     0.59018,      0.5923,     0.59644,     0.59618,     0.59591,     0.59564,     0.59538,     0.59511,     0.59484,     0.59458,     0.59431,     0.59404,
            0.59378,     0.59351,     0.59324,     0.59297,     0.59396,     0.59527,     0.59658,     0.59789,     0.59853,     0.59896,     0.59938,      0.5998,     0.60023,     0.60065,     0.60107,      0.6015,     0.60192,     0.60234,     0.60276,     0.60319,      0.6091,     0.60991,     0.61071,
            0.61151,     0.61232,     0.61312,     0.61392,     0.62135,     0.62883,      0.6325,     0.63602,     0.63973,     0.64371,     0.65122,     0.65206,     0.65291,     0.65375,      0.6546,     0.65544,     0.65629,     0.65693,     0.65716,     0.65739,     0.65762,     0.65784,     0.65807,
             0.6583,     0.65852,     0.65875,     0.65898,     0.65921,     0.65943,     0.65966,     0.65989,     0.66011,     0.66034,     0.66057,      0.6608,     0.66102,     0.66125,     0.66148,      0.6617,     0.66193,     0.66216,     0.66238,     0.66261,     0.66284,     0.66307,     0.66329,
            0.67016,      0.6704,     0.67063,     0.67087,     0.67111,     0.67134,     0.67158,     0.67182,     0.67205,     0.67229,     0.67253,     0.67276,       0.673,     0.67324,     0.67347,     0.67371,     0.67395,     0.67418,     0.67442,     0.67466,     0.67489,     0.67513,     0.67537,
             0.6756,     0.67584,     0.67608,     0.67631,     0.67655,     0.68497,     0.69104,     0.69156,     0.69208,     0.69259,     0.69311,     0.69363,     0.69414,     0.69466,     0.69518,     0.69569,     0.69621,     0.69672,     0.69724,     0.69776,     0.69747,     0.69682,     0.69617,
            0.69552,     0.69487,     0.69396,     0.69298,       0.692,      0.6991,     0.69948,     0.69985,     0.70022,      0.7006,     0.70097,     0.70134,     0.70172,     0.70209,     0.70246,     0.70284,     0.70321,     0.70358,     0.70395,     0.70433,      0.7047,     0.70507,     0.70545,
            0.70582,     0.70619,     0.71461,      0.7173,     0.71999,     0.72272,     0.72567,     0.72862,     0.73074,      0.7317,     0.73266,     0.73363,     0.73459,     0.73555,     0.73651,     0.73747,     0.73843,     0.74781,     0.74951,     0.75121,     0.75291,     0.75461,     0.75605,
            0.75684,     0.75763,     0.75843,     0.75922,     0.76002,     0.76081,     0.76161,      0.7624,     0.76319,     0.76399,     0.77389,     0.77471,     0.77552,     0.77634,     0.77716,     0.77798,      0.7788,     0.77961,     0.78043,     0.78125,     0.78207,     0.78289,     0.79349,
            0.79464,     0.79579,     0.79693,     0.79808,     0.79923,     0.80038,     0.80153,     0.80292,     0.80542,     0.80792,     0.81041,     0.81246,     0.81224,     0.81202,     0.81179,     0.81157,     0.81135,     0.81113,      0.8109,     0.81068,     0.81046,     0.81024,     0.80735,
            0.80668,     0.80601,     0.80534,     0.81036,     0.81607,     0.81768,     0.81928,     0.82089,      0.8225,      0.8241,     0.82571,     0.82663,     0.82654,     0.82645,     0.82636,     0.82627,     0.82618,     0.82609,     0.82599,      0.8259,     0.82581,     0.82572,     0.82563,
            0.82554,     0.82545,     0.82536,     0.82527,     0.82518,     0.82509,       0.825,     0.82491,     0.82482,     0.82472,     0.82463,     0.82454,     0.82445,     0.82436,     0.83535,     0.83491,     0.83447,     0.83402,     0.83358,     0.83233,     0.83539,     0.83845,     0.84151,
            0.84029,     0.83978,     0.83926,     0.83875,     0.83754,     0.83657,     0.83733,      0.8381,     0.83886,     0.83963,      0.8404,     0.84116,     0.84193,     0.84269,     0.84346,     0.84422,     0.84499,     0.84576,     0.84652,     0.84729,     0.84805,     0.86232,      0.8641,
            0.86589,     0.86768,     0.86947,     0.87126,     0.87305,     0.87483,     0.88948,     0.89026,     0.89104,     0.89182,      0.8926,     0.89339,     0.89417,     0.89495,     0.89573,     0.89651,     0.89729,     0.89807,     0.89885,     0.89963,     0.90041,     0.90119,     0.90197,
            0.90275,     0.90437,     0.90727,     0.91016,     0.91306,     0.91596,     0.91648,     0.91582,     0.91535,     0.91602,      0.9167,     0.91738,     0.91806,     0.91873,     0.91941,     0.92009,     0.92076,     0.92144,     0.92212,     0.92279,     0.92347,     0.92415,     0.92483,
             0.9255,     0.92618,     0.92686,     0.92753,     0.92821,     0.92889,     0.92956,     0.93024,     0.93092,     0.93061,     0.93009,     0.92949,     0.92881,     0.92847,     0.92832,     0.92818,     0.92803,     0.92788,     0.92773,     0.92758,     0.92743,     0.92728,     0.93225,
            0.93745,     0.94265,      0.9442,     0.94383,     0.94345,     0.94333,     0.94325,     0.94317,     0.94309,     0.94301,     0.94294,     0.94286,     0.94278,      0.9427,     0.94262,     0.94254,     0.94246,     0.94238,     0.94229,     0.94191,     0.94152,     0.94116,     0.94102,
            0.94087,     0.94072,     0.94058,     0.94043,     0.94029,     0.94014,        0.94,     0.93992,     0.93984,     0.93976,     0.93968,      0.9396,     0.93952,     0.93944,     0.93936,     0.93928,      0.9392,     0.93912,     0.93904,     0.93896,     0.93888,      0.9388,      0.9374,
            0.93725,     0.93711,     0.93696,     0.93682,     0.93667,     0.93652,     0.93638,     0.93623,     0.93473,     0.93464,     0.93455,     0.93446,     0.93436,     0.93427,     0.93418,     0.93409,       0.934,     0.93391,     0.93381,     0.93372,     0.93363,     0.93354,     0.93345,
            0.93336,     0.93168,     0.93149,      0.9313,     0.93111,     0.93092,     0.93073,     0.93054,     0.93035,     0.93015,     0.92995,     0.92974,     0.92953,     0.92933,     0.92912,     0.92892,     0.92871,     0.92675,      0.9265,     0.92625,     0.92601,     0.92576,     0.92551,
            0.92527,     0.92502,     0.92768,     0.93056,     0.93345,     0.93634,     0.93922,     0.94211,       0.945,     0.94788,     0.94825,     0.94758,     0.94556,     0.94499,     0.94435,     0.94271,     0.94191,     0.94139,     0.94411,     0.94682,     0.94953,     0.95225,     0.95496,
            0.95768,     0.96039,      0.9631,     0.96582,     0.96853,     0.96947,     0.96908,     0.96874,     0.96867,      0.9686,     0.96853,     0.96846,     0.96839,     0.96832,     0.96825,     0.96818,     0.96811,     0.96804,     0.96797,      0.9679,     0.96783,     0.96776,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1]]), 'Confidence', 'Precision'], [array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
          0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
          0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
          0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
          0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
           0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
           0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
           0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
           0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
           0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
           0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
           0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
           0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
           0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
           0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
           0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
           0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
           0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
           0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
           0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
           0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
            0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
           0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
           0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
           0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
            0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
           0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
           0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
           0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
            0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
           0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
           0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
           0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
           0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
           0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
           0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
           0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
           0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
           0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
           0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
           0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
           0.98498,     0.98599,     0.98699,     0.98799,     0.98899,     0.98999,     0.99099,     0.99199,     0.99299,     0.99399,     0.99499,       0.996,       0.997,       0.998,       0.999,           1]), array([[    0.95946,     0.95946,     0.95946,     0.95946,     0.95946,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,
            0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,
            0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94595,     0.94439,     0.93881,     0.93323,     0.93243,     0.93243,     0.93243,     0.93005,     0.92748,      0.9249,     0.92233,     0.91976,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,
            0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,
            0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,
            0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91892,     0.91874,     0.91773,     0.91672,     0.91571,      0.9147,     0.91369,     0.91268,     0.91167,     0.91066,     0.90965,
            0.90864,     0.90763,     0.90662,     0.90561,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,
            0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,
            0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,
            0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,
            0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,     0.90541,      0.9035,     0.90074,     0.89798,
            0.89522,     0.89246,     0.88866,     0.88457,     0.88049,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,
            0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,
            0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,
            0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87838,     0.87817,      0.8769,     0.87563,     0.87436,     0.87309,     0.87183,     0.87056,     0.86929,     0.86802,     0.86675,     0.86548,     0.84949,
            0.84587,     0.84225,     0.83862,     0.83784,     0.83784,     0.83784,     0.83784,     0.83784,     0.83784,     0.83784,     0.83784,     0.83763,      0.8371,     0.83658,     0.83606,     0.83553,     0.83501,     0.83448,     0.83396,     0.83344,     0.83291,     0.83239,     0.83187,
            0.83134,     0.83082,      0.8303,     0.82977,     0.82925,     0.82873,      0.8282,     0.82768,     0.82716,     0.82663,     0.82611,     0.82559,     0.82506,     0.82454,     0.82277,     0.82015,     0.81752,      0.8149,     0.81228,      0.7973,      0.7973,      0.7973,      0.7973,
            0.78212,     0.77915,     0.77618,     0.77321,     0.76636,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,
            0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,
            0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.75676,     0.74146,     0.73516,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,
            0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72973,     0.72494,     0.71915,     0.71259,     0.70524,     0.70168,     0.70013,     0.69858,     0.69702,     0.69547,     0.69392,     0.69236,     0.69081,     0.68926,     0.68919,
            0.68919,     0.68919,     0.68604,     0.68124,     0.67643,     0.67485,     0.67387,     0.67289,     0.67191,     0.67093,     0.66996,     0.66898,       0.668,     0.66702,     0.66604,     0.66506,     0.66408,      0.6631,     0.66198,     0.65738,     0.65278,     0.64848,      0.6468,
            0.64513,     0.64345,     0.64178,     0.64011,     0.63843,     0.63676,     0.63511,     0.63423,     0.63335,     0.63247,     0.63159,      0.6307,     0.62982,     0.62894,     0.62806,     0.62718,      0.6263,     0.62542,     0.62454,     0.62366,     0.62278,      0.6219,     0.60709,
            0.60561,     0.60413,     0.60264,     0.60116,     0.59967,     0.59819,     0.59671,     0.59522,     0.58059,     0.57974,     0.57888,     0.57803,     0.57718,     0.57632,     0.57547,     0.57462,     0.57376,     0.57291,     0.57206,      0.5712,     0.57035,      0.5695,     0.56865,
            0.56779,     0.55286,     0.55124,     0.54963,     0.54801,     0.54639,     0.54477,     0.54316,     0.54154,      0.5399,     0.53822,     0.53654,     0.53486,     0.53318,     0.53151,     0.52983,     0.52815,     0.51291,     0.51108,     0.50926,     0.50743,     0.50561,     0.50378,
            0.50196,     0.50013,         0.5,         0.5,         0.5,         0.5,         0.5,         0.5,         0.5,         0.5,     0.49527,     0.48863,     0.46949,     0.46434,     0.45864,     0.44473,     0.43833,     0.43243,     0.43243,     0.43243,     0.43243,     0.43243,     0.43243,
            0.43243,     0.43243,     0.43243,     0.43243,     0.43243,     0.42922,     0.42359,     0.41876,     0.41782,     0.41688,     0.41594,       0.415,     0.41407,     0.41313,     0.41219,     0.41125,     0.41031,     0.40937,     0.40844,      0.4075,     0.40656,     0.40562,     0.40455,
            0.40345,     0.40234,     0.40124,     0.40013,     0.39903,     0.39792,     0.39682,     0.39571,     0.39461,      0.3935,      0.3924,      0.3792,       0.351,     0.35064,     0.35027,     0.34991,     0.34955,     0.34919,     0.34882,     0.34846,      0.3481,     0.34773,     0.34737,
            0.34701,     0.34664,     0.34628,     0.34592,     0.34555,     0.34519,     0.34483,     0.34447,      0.3441,     0.34374,     0.34338,     0.34301,     0.34265,     0.34229,     0.34192,     0.34156,      0.3412,     0.34083,     0.34047,     0.34011,     0.33975,     0.33938,     0.33902,
            0.33866,     0.33829,     0.33793,     0.32218,     0.31931,     0.31644,     0.31357,     0.31076,     0.30955,     0.30833,     0.30711,      0.3059,     0.30468,     0.30346,     0.30225,     0.30103,     0.29981,      0.2986,     0.29738,     0.29531,     0.29318,     0.29105,     0.28891,
            0.28678,     0.28465,     0.28329,     0.28245,     0.28162,     0.28078,     0.27995,     0.27911,     0.27827,     0.27744,      0.2766,     0.27577,     0.27493,      0.2741,     0.27326,     0.27243,     0.27159,     0.27076,     0.26929,     0.26693,     0.26457,     0.26221,     0.25985,
            0.25749,     0.24292,     0.24214,     0.24136,     0.24058,     0.23981,     0.23903,     0.23825,     0.23747,      0.2367,     0.23592,     0.23514,     0.23436,     0.23359,     0.23281,     0.23203,     0.23125,     0.23048,     0.22947,     0.22328,     0.21709,     0.21235,     0.20785,
            0.20334,     0.18866,     0.18805,     0.18743,     0.18682,     0.18621,     0.18559,     0.18498,     0.18436,     0.18375,     0.18314,     0.18252,     0.18191,     0.18129,     0.18068,     0.18007,     0.17945,     0.17884,     0.17822,     0.17761,       0.177,     0.17638,     0.17577,
            0.17516,     0.17456,     0.17395,     0.17335,     0.17274,     0.17214,     0.17153,     0.17093,     0.17032,     0.16972,     0.16911,     0.16851,      0.1679,      0.1673,     0.16669,     0.16609,     0.16548,     0.16488,     0.16427,     0.16367,     0.16306,     0.16246,     0.14787,
            0.14634,     0.14481,     0.14328,     0.14175,     0.14022,     0.13869,     0.13716,     0.13563,     0.13482,     0.13435,     0.13388,     0.13341,     0.13294,     0.13247,       0.132,     0.13153,     0.13106,     0.13059,     0.13012,     0.12965,     0.12918,     0.12871,     0.12825,
            0.12778,     0.12731,     0.12684,     0.12637,      0.1259,     0.12543,     0.12496,     0.12449,     0.12402,     0.12355,     0.12308,     0.12261,     0.12214,     0.12167,     0.11492,     0.10806,     0.10759,     0.10711,     0.10664,     0.10616,     0.10569,     0.10522,     0.10474,
            0.10427,     0.10379,     0.10332,     0.10284,     0.10237,     0.10189,     0.10142,     0.10094,     0.10047,    0.099995,     0.09952,    0.099046,    0.098571,    0.098097,    0.097622,    0.097147,    0.096673,    0.096198,    0.095724,    0.095249,    0.094774,    0.093914,    0.092818,
           0.091721,    0.090625,    0.089529,    0.088433,    0.087336,     0.08624,    0.085144,    0.084047,    0.082951,    0.081855,    0.078793,    0.071012,    0.065959,    0.063072,    0.060186,      0.0573,    0.054413,    0.053669,    0.053228,    0.052788,    0.052348,    0.051907,    0.051467,
           0.051026,    0.050586,    0.050146,    0.049705,    0.049265,    0.048825,    0.048384,    0.047944,    0.047504,    0.047063,    0.046623,    0.046183,    0.045742,    0.045302,    0.044862,    0.044421,    0.043981,    0.043541,      0.0431,     0.04266,     0.04222,    0.041779,    0.041339,
           0.040899,    0.026286,    0.022326,    0.018366,    0.014405,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,
                  0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,
                  0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,
                  0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,
                  0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0]]), 'Confidence', 'Recall']]
fitness: 0.4703156070791159
keys: ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
maps: array([    0.42494])
names: {0: 'Durham College'}
plot: True
results_dict: {'metrics/precision(B)': 0.8102350935919751, 'metrics/recall(B)': 0.8654826950462009, 'metrics/mAP50(B)': 0.8787032136015919, 'metrics/mAP50-95(B)': 0.4249392063543963, 'fitness': 0.4703156070791159}
save_dir: PosixPath('/Users/mainfolder/Yolo_Label/runs/detect/train11')
speed: {'preprocess': 0.7252243981845137, 'inference': 102.94479563616324, 'loss': 4.146410071331522e-05, 'postprocess': 0.224562658779863}
task: 'detect'























































