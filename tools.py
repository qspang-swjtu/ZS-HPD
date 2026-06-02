import torch
import torch.nn.functional as F
import time
import torchvision.transforms as transforms
import kornia



def content_aware_downsample_larger_region(image_tensor,
                                           compare_kernel_size=5):
    _, C, _, _ = image_tensor.shape
    device = image_tensor.device
    padding = compare_kernel_size // 2

    # #  梯度幅度 
    if C > 1: img_gray = transforms.functional.rgb_to_grayscale(image_tensor)
    else: img_gray = image_tensor
    # shape : [H , W]
    
    # 计算每个像素的梯度
    spatial_grads = kornia.filters.spatial_gradient(img_gray, order=1)
    tensor_sy = spatial_grads[:, :, 0, :, :]  # shape: (B, 1, H, W)
    tensor_sx = spatial_grads[:, :, 1, :, :]  # shape: (B, 1, H, W)
    
    # shape : [H ,W]
    tensor_sx = tensor_sx.to(device)
    tensor_sy = tensor_sy.to(device)
    # 用于合并梯度的全1卷积核
    sum_sobel = torch.ones(1,
                           1,
                           compare_kernel_size,
                           compare_kernel_size,
                           device=device)
    # 合并梯度
    sum_sx = F.conv2d(tensor_sx, sum_sobel, padding=padding)
    sum_sy = F.conv2d(tensor_sy, sum_sobel, padding=padding)
    # 计算幅度
    gradient_magnitude = torch.hypot(sum_sx, sum_sy)

    metric_map = gradient_magnitude.to(device)  # [1, 1, H, W]

    region_metric = metric_map

    # --- 提取四个子网格的像素值和区域度量值 ---
    val_00 = image_tensor[:, :, ::2, ::2]
    val_11 = image_tensor[:, :, 1::2, 1::2]
    val_01 = image_tensor[:, :, ::2, 1::2]
    val_10 = image_tensor[:, :, 1::2, ::2]
    # (0,0)
    metric_00 = region_metric[:, :, ::2, ::2]
    # (1,1)
    metric_11 = region_metric[:, :, 1::2, 1::2]
    # (0,1)
    metric_01 = region_metric[:, :, ::2, 1::2]
    # (1,0)
    metric_10 = region_metric[:, :, 1::2, ::2]

    # --- 生成 D1 ---
    # 比较 (0,0) 区域和 (1,1) 区域的度量，选择度量大的对应中心像素
    mask_00_region_gt_11 = metric_00 >= metric_11
    D1 = torch.where(mask_00_region_gt_11.expand_as(val_00), val_00, val_11)
    D3 = torch.where(mask_00_region_gt_11.expand_as(val_00),val_11,val_00)
    # --- 生成 D2 ---
    # 比较 (0,1) 区域和 (1,0) 区域的度量，选择度量大的对应中心像素
    mask_01_region_gt_10 = metric_01 >= metric_10
    D2 = torch.where(mask_01_region_gt_10.expand_as(val_01), val_01, val_10)
    D4 = torch.where(mask_01_region_gt_10.expand_as(val_01),val_10,val_01)
    return D1, D2 ,D3 ,D4 

def build_random_blank(
        image_batch_tensor,
        patch_size=7,
        M=10,
        
        num_random_candidates_per_anchor=1024,
        similarity_metric='l2',
        anchor_processing_batch_size=2048,
        sigma = 25,

        device=None):
    
    start_time = time.time()
    
    if device is None:
        device = image_batch_tensor.device
    compute_dtype = torch.float16 
    image_batch_tensor = image_batch_tensor.to(dtype=compute_dtype)
    
    B, C, H, W = image_batch_tensor.shape
    N_pixels_per_image = H * W
    D_patch = C * patch_size * patch_size

    padding = patch_size // 2
    
    ycbr_image_tensor = rgb_to_ycbr(image_batch_tensor.to(dtype=torch.float16))
    y_channel_tensor = ycbr_image_tensor[:, 0:1, :, :] # 形状为 (B, 1, H, W)
    
    unfolded_patches_y = F.unfold(y_channel_tensor, 
                                  kernel_size=patch_size, 
                                  padding=padding, 
                                  stride=1)
    
    anchor_patches_flat_full = unfolded_patches_y.permute(
        0, 2, 1).contiguous().to(device)  # (B, N_pixels, D_patch)

    r_coords, c_coords = torch.meshgrid(torch.arange(H, device=device),
                                        torch.arange(W, device=device),
                                        indexing='ij')
    anchor_coords_per_image = torch.stack((r_coords.ravel(), c_coords.ravel()),
                                          dim=-1)
    anchor_coords_batch_full = anchor_coords_per_image.unsqueeze(0).expand(
        B, -1, -1).to(device)  # (B, N_pixels, 2)
    # --- 结束预计算 ---

    final_pixel_bank_coords = torch.zeros(B,
                                          N_pixels_per_image,
                                          M,
                                          2,
                                          dtype=torch.long,
                                          device=device)

    
    num_anchor_batches = (N_pixels_per_image + anchor_processing_batch_size -
                          1) // anchor_processing_batch_size

    for i in range(num_anchor_batches):
        start_idx = i * anchor_processing_batch_size
        end_idx = min((i + 1) * anchor_processing_batch_size,
                      N_pixels_per_image)

        current_anchors_in_sub_batch = end_idx - start_idx
        if current_anchors_in_sub_batch == 0:
            continue

        #  提取当前子批次的锚点块
        # current_anchor_patches_sub_batch: (B, current_anchors_in_sub_batch, D_patch)
        current_anchor_patches_sub_batch = anchor_patches_flat_full[:,
                                                                    start_idx:
                                                                    end_idx, :]
        current_anchor_patches_sub_batch_coords = anchor_coords_batch_full[:,
                                                                           start_idx:
                                                                           end_idx, :]
    
        
        #  为当前子批次的锚点生成随机候选索引
        anchor_r_expanded = current_anchor_patches_sub_batch_coords[
            ..., 0].unsqueeze(-1)  # 假设你有这个坐标张量
        anchor_c_expanded = current_anchor_patches_sub_batch_coords[
            ..., 1].unsqueeze(-1)

        
        offsets_r_sub = torch.normal(mean=0.0,
                                     std=sigma,
                                     size=(B, current_anchors_in_sub_batch,
                                           num_random_candidates_per_anchor),
                                     device=device)
       
        offsets_c_sub = torch.normal(mean=0.0,
                                     std=sigma,
                                     size=(B, current_anchors_in_sub_batch,
                                           num_random_candidates_per_anchor),
                                     device=device)
        
        cand_r_sub = torch.round(anchor_r_expanded + offsets_r_sub).long()
        cand_c_sub = torch.round(anchor_c_expanded + offsets_c_sub).long()

        cand_r_sub = torch.clamp(cand_r_sub, 0, H - 1)
        cand_c_sub = torch.clamp(cand_c_sub, 0, W - 1)

        # 转换回扁平化索引
        random_candidate_indices_sub = cand_r_sub * W + cand_c_sub

        #  提取这些候选块 (从完整的 anchor_patches_flat_full 中提取)
        # candidate_patches_sub: (B, current_anchors_in_sub_batch, num_random_candidates_per_anchor, D_patch)

        # batch_indices_for_gather: (B, 1, 1) for broadcasting with random_candidate_indices_sub
        batch_indices_for_gather = torch.arange(B, device=device).view(B, 1, 1)
        # 这里的 anchor_patches_flat_full 已经是 (B, N_pixels, D_patch)
        # random_candidate_indices_sub 已经是 (B, current_anchors, num_cand)
        # 我们希望 candidate_patches_sub[b, anchor_idx_in_sub, k, :] = anchor_patches_flat_full[b, random_candidate_indices_sub[b, anchor_idx_in_sub, k], :]
        candidate_patches_sub = anchor_patches_flat_full[
            batch_indices_for_gather,  # Selects the batch
            random_candidate_indices_sub,  # Selects the candidate pixels for each anchor in the sub_batch
            :  # Selects all features D_patch
        ]

        #  提取这些候选坐标 (从完整的 anchor_coords_batch_full 中提取)
        # candidate_coords_sub: (B, current_anchors_in_sub_batch, num_random_candidates_per_anchor, 2)
        candidate_coords_sub = anchor_coords_batch_full[
            batch_indices_for_gather,
            random_candidate_indices_sub,
            :  # Selects both r and c coordinates
        ]

        # 计算距离
        # diff_sub: (B, current_anchors_in_sub_batch, num_random_candidates_per_anchor, D_patch)
        diff_sub = candidate_patches_sub - current_anchor_patches_sub_batch.unsqueeze(
            2)

        if similarity_metric == 'l2':
            all_distances_sub = torch.norm(diff_sub, p=2, dim=3)
        elif similarity_metric == 'l1':
            all_distances_sub = torch.norm(diff_sub, p=1, dim=3)
        elif 'cdist' in similarity_metric:
            p_val = 2.0 if similarity_metric == 'l2_cdist' else 1.0
            # Reshape for cdist:
            # current_anchor_patches_reshaped: (B * current_anchors_in_sub_batch, 1, D_patch)
            # candidate_patches_sub_reshaped: (B * current_anchors_in_sub_batch, num_random_candidates_per_anchor, D_patch)
            bsb = current_anchor_patches_sub_batch.shape[
                0] * current_anchor_patches_sub_batch.shape[
                    1]  # B * current_anchors_in_sub_batch

            current_anchor_patches_reshaped = current_anchor_patches_sub_batch.reshape(
                bsb, 1, D_patch)
            candidate_patches_sub_reshaped = candidate_patches_sub.reshape(
                bsb, num_random_candidates_per_anchor, D_patch)

            distances_reshaped = torch.cdist(current_anchor_patches_reshaped,
                                             candidate_patches_sub_reshaped,
                                             p=p_val).squeeze(1)
            all_distances_sub = distances_reshaped.view(
                B, current_anchors_in_sub_batch,
                num_random_candidates_per_anchor)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")

        # Top-M 选择
        # top_m_indices_sub: (B, current_anchors_in_sub_batch, M)
        _, top_m_indices_sub = torch.topk(all_distances_sub,
                                          k=M,
                                          dim=2,
                                          largest=False)

        #  Gather 最终的坐标
        # index_expanded_sub: (B, current_anchors_in_sub_batch, M, 2)
        index_expanded_sub = top_m_indices_sub.unsqueeze(-1).expand(
            -1, -1, -1, 2)
        gathered_coords_sub = torch.gather(
            candidate_coords_sub,
            dim=
            2,  # Gather along the 'num_random_candidates_per_anchor' dimension
            index=index_expanded_sub)

        #  将当前子批次的结果存入最终的 pixel bank (扁平化的索引)
        final_pixel_bank_coords[:,
                                start_idx:end_idx, :, :] = gathered_coords_sub

        del current_anchor_patches_sub_batch, random_candidate_indices_sub, candidate_patches_sub, candidate_coords_sub, diff_sub, all_distances_sub, top_m_indices_sub, index_expanded_sub, gathered_coords_sub
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Reshape 最终结果
    pixel_bank_indices_coords_batch_reshaped = final_pixel_bank_coords.reshape(
        B, H, W, M, 2)
    print(f'共消耗时间 :{time.time() - start_time :.4f}')
    return pixel_bank_indices_coords_batch_reshaped


def random_noisy(
        image_batch_tensor,
        pixel_bank_coords_reshaped,
        M=10,
        device=None):
    if device is None:
        device = image_batch_tensor.device
    
    B, C, H, W = image_batch_tensor.shape

    m_indices1 = torch.randint(0,
                               M,
                               size=(B, H, W),
                               device=device,
                               dtype=torch.long)
    m_indices2 = torch.randint(0,
                               M,
                               size=(B, H, W),
                               device=device,
                               dtype=torch.long)

    conflict_mask = (m_indices1 == m_indices2)
    while torch.any(conflict_mask):
        num_conflicts = torch.sum(conflict_mask)
        m_indices2[conflict_mask] = torch.randint(0,
                                                  M,
                                                  size=(num_conflicts, ),
                                                  device=device,
                                                  dtype=torch.long)
        conflict_mask = (m_indices1 == m_indices2)

    m_indices_pair = torch.stack((m_indices1, m_indices2), dim=-1)  # (B,H,W,2)

    # Gather 选定的相似像素的坐标
    # m_indices_pair_expanded: (B,H,W,2,2) -> 最后一个2是(r,c)坐标
    m_indices_pair_expanded = m_indices_pair.unsqueeze(-1).expand(
        -1, -1, -1, -1, 2)

    # selected_similar_coords_pair: (B,H,W,2,2)
    # dim=3 对应原始 pixel_bank_coords_reshaped 中的 M 维度
    selected_similar_coords_pair = torch.gather(pixel_bank_coords_reshaped,
                                                dim=3,
                                                index=m_indices_pair_expanded)

    pseudo_images = []
    image_batch_permuted_flat = image_batch_tensor.permute(0, 2, 3, 1).reshape(
        B, H * W, C)

    for i in range(2):  # 为伪图1和伪图2
        current_selected_coords = selected_similar_coords_pair[:, :, :,
                                                               i, :]  # (B, H, W, 2)
        selected_sr_flat = current_selected_coords[...,
                                                   0].reshape(B,
                                                              -1)  # (B, H*W)
        selected_sc_flat = current_selected_coords[...,
                                                   1].reshape(B,
                                                              -1)  # (B, H*W)

        indices_for_gather = selected_sr_flat * W + selected_sc_flat  # (B, H*W)
        indices_for_gather_expanded = indices_for_gather.unsqueeze(-1).expand(
            -1, -1, C)  # (B, H*W, C)

        pseudo_image_flat = torch.gather(image_batch_permuted_flat,
                                         dim=1,
                                         index=indices_for_gather_expanded)
        pseudo_img = pseudo_image_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        pseudo_images.append(pseudo_img)
    return pseudo_images[0] ,pseudo_images[1]
    
def rgb_to_ycbr(image_tensor: torch.Tensor) -> torch.Tensor:

    if image_tensor.shape[1] != 3:
        return image_tensor
        raise ValueError("Input tensor must have 3 channels (RGB)")
    
    # 提取 R, G, B 通道
    r, g, b = image_tensor.split(1, dim=1)
    
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b
    
    return torch.cat([y, cb, cr], dim=1)

