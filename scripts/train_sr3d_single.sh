TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port 7777 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /root/vg_data/ \
    --val_freq 1 --batch_size 10 --save_freq 1 --print_freq 500 \
    --lr_backbone=2e-4 --lr=1e-4 \
    --dataset sr3d --test_dataset sr3d \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir /root/autodl-tmp/logs/ \
    --lr_decay_epochs 10 30 \
    --self_attend \
    --checkpoint_path /root/autodl-tmp/ckpt/PV-Ground_NR3D.pth \
    --max_epoch 140 \
    --model PVGround \
    --exp PVGround \
    # --pp_checkpoint /home/sjp/3DVG/vg_data/gf_detector_l6o256.pth \