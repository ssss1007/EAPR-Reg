import os
import glob
import sys
import random
import time
import pickle
import torch
import numpy as np
from checkpoints import load_registration_weights
# import scipy.ndimage
from argparse import ArgumentParser

try:
    from torchsummary import summary as torch_summary
except Exception:
    torch_summary = None

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm


VOI_LABELS = np.array([
    2, 3, 4, 7, 8, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 24, 28, 31, 41,
    42, 43, 46, 47, 49, 50, 51, 52,
    53, 54, 60, 63,
], dtype=np.int32)


def _to_bcdhw(arr, device):
    """
    Convert numpy batch from generator to torch [B, C, S1, S2, S3],
    while preserving the original spatial axis order from the loader.
    Supports:
      [B,H,W,D]
      [B,H,W,D,1]
      [B,1,H,W,D]
      [B,2,H,W,D,1] -> keep first component
    """
    t = torch.from_numpy(arr).to(device).float()
    if t.ndim == 6:
        t = t[:, 0, ...]
    if t.ndim == 4:
        t = t.unsqueeze(-1)
    if t.ndim != 5:
        raise RuntimeError(f'Unexpected input ndim={t.ndim}, shape={tuple(t.shape)}')

    # channel-last: [B,S1,S2,S3,C] -> [B,C,S1,S2,S3]
    if t.shape[-1] <= 4:
        return t.permute(0, 4, 1, 2, 3).contiguous()
    # channel-first already: [B,C,S1,S2,S3]
    if t.shape[1] <= 4:
        return t.contiguous()
    raise RuntimeError(f'Cannot infer channel axis for shape={tuple(t.shape)}')


def _to_dhw(arr):
    vol = np.asarray(arr)
    vol = np.squeeze(vol)
    if vol.ndim != 3:
        raise RuntimeError(f'Expected 3D volume, got shape={vol.shape}')
    # Align with the APE pretraining convention for IXI pkl: HWD -> DHW
    if vol.shape[2] >= vol.shape[0] and vol.shape[2] >= vol.shape[1]:
        vol = np.transpose(vol, (2, 0, 1))
    return vol


def _load_pkl_img_seg(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        img, seg = obj[0], obj[1]
    elif isinstance(obj, dict):
        img = obj.get('image', obj.get('img', obj.get('vol', obj.get('data', None))))
        seg = obj.get('seg', obj.get('label', obj.get('mask', None)))
        if img is None or seg is None:
            raise RuntimeError(f'Cannot parse image/seg from dict pkl: {path}')
    else:
        raise RuntimeError(f'Unsupported pkl payload type for validation: {type(obj)}')
    return _to_dhw(img), _to_dhw(seg)


def _eval_val_dice_pkl(model, val_files, device, num_pairs=20, atlas_file=None):
    if not val_files:
        return None
    model.eval()
    dices = []

    atlas_img = atlas_seg = None
    if atlas_file:
        atlas_img, atlas_seg = _load_pkl_img_seg(atlas_file)

    with torch.no_grad():
        if atlas_file:
            val_indices = range(len(val_files))
        else:
            val_indices = range(min(num_pairs, len(val_files)))

        for _ in val_indices:
            if atlas_file:
                j = _
                fixed_img, fixed_seg = atlas_img, atlas_seg
                moving_img, moving_seg = _load_pkl_img_seg(val_files[j])
            else:
                if len(val_files) > 1:
                    i, j = random.sample(range(len(val_files)), 2)
                else:
                    i = j = 0
                fixed_img, fixed_seg = _load_pkl_img_seg(val_files[i])
                moving_img, moving_seg = _load_pkl_img_seg(val_files[j])

            fixed = torch.from_numpy(fixed_img).float().unsqueeze(0).unsqueeze(0).to(device)
            moving = torch.from_numpy(moving_img).float().unsqueeze(0).unsqueeze(0).to(device)
            moving_seg_t = torch.from_numpy(moving_seg).float().unsqueeze(0).unsqueeze(0).to(device)

            pred = model(moving, fixed)
            flow = pred[1]

            trf = vxm.layers.SpatialTransformer(fixed.shape[2:], mode='nearest').to(device)
            warped_seg = trf(moving_seg_t, flow).detach().cpu().numpy().squeeze()
            warped_seg = np.rint(warped_seg).astype(np.int32)
            fixed_seg = fixed_seg.astype(np.int32)

            overlap = vxm.py.utils.dice(warped_seg, fixed_seg, labels=VOI_LABELS)
            dices.append(float(np.mean(overlap)))

    if len(dices) == 0:
        return None
    return float(np.mean(dices))


def train(datadir,
          model_dir,
          load_model,
          gpu,
          initial_epoch,
          epochs,
          steps_per_epoch,
          batch_size,
          first_channel=8,
          atlas=False,
          atlas_file=None,
          bidir=False,
          use_ape=False,
          ape_ckpt=None,
          sam2_cfg=None,
          sam2_ckpt=None,
          sam2_repo=None,
          ape_k_list='24,10,6',
          ape_temp=0.3,
          ape_channels=0,
          use_raw_medsam=False,
          ape_use_raw_fusion=False,
          ape_use_gated_raw_fusion=False,
          ape_use_eadf=False,
          ape_use_half_refine=False,
          ape_use_context_refine=False,
          ape_use_proto_attn=False,
          raw_medsam_use_refine=False,
          show_summary=False,
          val_dir=None,
          val_atlas_file=None,
          val_every=1,
          val_pairs=20,
          lr=1e-4,
          image_loss_weight=1.0,
          grad_weight=1.0,
          flow_mag_weight=0.0):

    # Support nested dataset folders and common medical volume extensions.
    patterns = ('*.nii.gz', '*.nii', '*.mgz', '*.npz', '*.pkl')
    train_vol_names = []
    for pat in patterns:
        train_vol_names.extend(glob.glob(os.path.join(datadir, pat)))
        train_vol_names.extend(glob.glob(os.path.join(datadir, '**', pat), recursive=True))
    train_vol_names = sorted(set(train_vol_names))
    random.shuffle(train_vol_names)  # shuffle volume list
    assert len(train_vol_names) > 0, (
        f'Could not find any training data in {datadir}. '
        'Expected files like .nii.gz/.nii/.mgz/.npz/.pkl (also searched recursively).'
    )

    val_files = []
    if val_dir:
        val_files = sorted(glob.glob(os.path.join(val_dir, '*.pkl')))
        val_files += sorted(glob.glob(os.path.join(val_dir, '**', '*.pkl'), recursive=True))
        val_files = sorted(set(val_files))
        if len(val_files) == 0:
            print(f'[Warn] val_dir set but no .pkl found in {val_dir}, skip val dice.')
        else:
            print(f'[Info] Validation cases: {len(val_files)}')

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = True

    generator_kwargs = dict(add_feat_axis=add_feat_axis)
    if atlas_file:
        generator = vxm.generators.scan_to_atlas(
            train_vol_names,
            atlas_file=atlas_file,
            batch_size=batch_size,
            bidir=bidir,
            **generator_kwargs,
        )
    else:
        generator = vxm.generators.scan_to_scan(
            train_vol_names,
            batch_size=batch_size,
            bidir=bidir,
            **generator_kwargs,
        )

    # extract shape from sampled input after canonical conversion
    sample_inputs, _ = next(generator)
    sample_source = _to_bcdhw(sample_inputs[0], device='cpu')
    inshape = tuple(int(v) for v in sample_source.shape[2:])

    os.makedirs(model_dir, exist_ok=True)


    # prepare odel folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    torch.backends.cudnn.deterministic = True

    # prepare the model
    ape_k_values = tuple(int(v.strip()) for v in str(ape_k_list).split(',') if v.strip())
    ape_channels_value = None if int(ape_channels) <= 0 else int(ape_channels)
    model = vxm.iirp.RPNet(
        inshape,
        first_channel=int(first_channel),
        use_ape=use_ape,
        ape_ckpt=ape_ckpt,
        sam2_cfg=sam2_cfg,
        sam2_ckpt=sam2_ckpt,
        sam2_repo=sam2_repo,
        ape_k_list=ape_k_values,
        ape_temp=ape_temp,
        ape_channels=ape_channels_value,
        use_raw_medsam=use_raw_medsam,
        ape_use_raw_fusion=ape_use_raw_fusion,
        ape_use_gated_raw_fusion=ape_use_gated_raw_fusion,
        ape_use_eadf=ape_use_eadf,
        ape_use_half_refine=ape_use_half_refine,
        ape_use_context_refine=ape_use_context_refine,
        ape_use_proto_attn=ape_use_proto_attn,
        raw_medsam_use_refine=raw_medsam_use_refine,
    )
    model.to(device)
    if show_summary:
        if torch_summary is None:
            print('[Warn] torchsummary not available, skip model summary.')
        else:
            try:
                torch_summary(model, input_size=[(1, *inshape), (1, *inshape)])
            except Exception as exc:
                print(f'[Warn] torchsummary failed ({exc}), continue without summary.')
    if load_model != False:
        print('loading', load_model)
        best_model = torch.load(load_model, map_location=device)
        load_registration_weights(model, best_model)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    
    # prepare losses
    Losses = [vxm.losses.NCC().loss, vxm.losses.Grad_2('l2').loss]
    Weights = [float(image_loss_weight), float(grad_weight)]

    # training/validate loops
    best_val_dice = -float('inf')
    for epoch in range(initial_epoch, epochs):
        start_time = time.time()
        
        # training
        model.train()
        train_losses = []
        train_total_loss = []
        for step in range(steps_per_epoch):
            
            # generate inputs (and true outputs) and convert them to tensors
            inputs, labels = next(generator)
            # inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
            # labels = [torch.from_numpy(d).to(device).float() for d in labels]
            inputs = [_to_bcdhw(d, device) for d in inputs]
            labels = [_to_bcdhw(d, device) for d in labels]
            source = inputs[0]
            target = inputs[1]
            # run inputs through the model to produce a warped image and flow field
            pred = model(source, target)

            # calculate total loss
            loss = 0
            loss_list = []
            for i, Loss in enumerate(Losses):
                curr_loss = Loss(pred[i], target) * Weights[i]
                loss_list.append(curr_loss.item())
                loss += curr_loss
            if flow_mag_weight and flow_mag_weight > 0:
                mag_loss = torch.mean(pred[1] ** 2) * float(flow_mag_weight)
                loss_list.append(mag_loss.item())
                loss += mag_loss
            train_losses.append(loss_list)
            train_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, epochs)
        time_info = 'Total %.2f sec' % (time.time() - start_time)
        train_losses = ', '.join(['%.4f' % f for f in np.mean(train_losses, axis=0)])
        train_loss_info = 'Train loss: %.4f  (%s)' % (np.mean(train_total_loss), train_losses)
        print(' - '.join((epoch_info, time_info, train_loss_info)), flush=True)

        val_dice = None
        if val_files and ((epoch + 1) % int(max(1, val_every)) == 0):
            eval_atlas_file = val_atlas_file or atlas_file
            val_dice = _eval_val_dice_pkl(
                model,
                val_files,
                device=device,
                num_pairs=int(max(1, val_pairs)),
                atlas_file=eval_atlas_file,
            )
            if val_dice is None:
                print(f'Epoch {epoch + 1} - Val Dice: N/A (no valid labels)')
            else:
                print(f'Epoch {epoch + 1} - Val Dice: {val_dice:.4f}')
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    torch.save(model.state_dict(), os.path.join(model_dir, 'best_val.pt'))
                    print(f'Epoch {epoch + 1} - Saved best_val.pt ({best_val_dice:.4f})')
    
        # save model checkpoint every epoch
        torch.save(model.state_dict(), os.path.join(model_dir, '%04d.pt' % (epoch+1)))
    

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('datadir', help='base data directory')
    parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
    parser.add_argument('--load-model', default=False, help='optional model file to initialize with')
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
    parser.add_argument('--first-channel', type=int, default=8, help='base channel width for the RPNet CNN branch')
    parser.add_argument('--use-ape', action='store_true', help='enable pretrained APE anatomy guidance')
    parser.add_argument('--ape-ckpt', default=None, help='path to APE checkpoint, e.g. ape_pretrained_plan_d_best.pth')
    parser.add_argument('--sam2-cfg', default=None, help='path to sam2 config yaml')
    parser.add_argument('--sam2-ckpt', default=None, help='path to MedSAM2 checkpoint')
    parser.add_argument('--sam2-repo', default=None, help='path containing sam2main package')
    parser.add_argument('--ape-k-list', default='24,10,6', help='comma-separated prototype counts, e.g. 24,10,6')
    parser.add_argument('--ape-temp', type=float, default=0.3, help='APE temperature used for guidance head')
    parser.add_argument('--ape-channels', type=int, default=0, help='APE feature channels. 0 auto-infers from checkpoint; use 128 for 128-channel APE')
    parser.add_argument('--use-raw-medsam', action='store_true', help='use frozen MedSAM2 raw FPN features projected to 64 channels without APE prototypes')
    parser.add_argument('--ape-use-raw-fusion', action='store_true', help='fuse APE prototype features with projected raw MedSAM2 features')
    parser.add_argument('--ape-use-gated-raw-fusion', action='store_true', help='gate projected raw MedSAM2 features as residuals to APE prototype features')
    parser.add_argument('--ape-use-eadf', action='store_true', help='entropy-aware fusion of APE reconstructed features with projected raw features')
    parser.add_argument('--ape-use-half-refine', action='store_true', help='add an H/2 image refinement stage after the APE pyramid')
    parser.add_argument('--ape-use-context-refine', action='store_true', help='inject upsampled APE context into full-resolution image refinement')
    parser.add_argument('--ape-use-proto-attn', action='store_true', help='enhance APE features with lightweight voxel-to-prototype attention')
    parser.add_argument('--raw-medsam-use-refine', action='store_true', help='add refine32 to the raw MedSAM baseline')
    parser.add_argument('--show-summary', action='store_true', help='print torchsummary before training')
    parser.add_argument('--atlas-file', default=None, help='optional fixed atlas pkl/volume path for atlas-based training')
    parser.add_argument('--val-dir', default=None, help='optional validation directory (.pkl with [img,seg])')
    parser.add_argument('--val-atlas-file', default=None, help='optional fixed atlas pkl path for atlas-based validation (defaults to --atlas-file)')
    parser.add_argument('--val-every', type=int, default=1, help='evaluate validation Dice every N epochs')
    parser.add_argument('--val-pairs', type=int, default=20, help='number of random validation pairs per eval')
    parser.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--image-loss-weight', type=float, default=1.0, help='weight for image similarity loss (NCC)')
    parser.add_argument('--grad-weight', type=float, default=1.0, help='weight for displacement smoothness loss (Grad)')
    parser.add_argument('--flow-mag-weight', type=float, default=0.0, help='optional L2 penalty on displacement magnitude')

    args = parser.parse_args()
    train(**vars(args))
