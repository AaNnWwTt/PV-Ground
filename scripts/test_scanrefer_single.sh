TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port 1111 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /root/autodl-tmp/vg_data/ \
    --val_freq 3 --batch_size 24 --save_freq 3 --print_freq 500 \
    --lr_backbone=2e-3 --lr=2e-4 \
    --dataset scanrefer --test_dataset scanrefer \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir /root/autodl-tmp/logs/ \
    --lr_decay_epochs 50 75 \
    --self_attend --augment_det \
    --model PVGround \
    --eval
    # --checkpoint_path /root/autodl-tmp/PV-Ground_ScanRefer_single_finetune.pth \