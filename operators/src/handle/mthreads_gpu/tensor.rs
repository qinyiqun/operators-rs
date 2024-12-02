use crate::handle::mthreads_gpu::opbindings::{
    infiniopTensorDescriptor_t, DataLayout, TensorDescriptor,
};
use crate::{get_static, infiniops, LaunchError, TensorLayout};
use digit_layout::{
    types::{F16, F32, U32, U64},
    DigitLayout,
};

pub fn get_shape_stride(tensor_layout: &TensorLayout) -> Result<(Vec<u64>, Vec<i64>), LaunchError> {
    let ndim = tensor_layout.ndim();
    let unit = tensor_layout.dt().nbytes() as isize;
    let mut shape: Vec<u64> = vec![];
    let mut stride: Vec<i64> = vec![];
    match ndim {
        1 => {
            let &[x] = tensor_layout.shape() else {
                unreachable!()
            };
            let &[x_s] = tensor_layout.strides() else {
                unreachable!()
            };

            get_static! {
                x
                x_s
            }

            let mut shape_ = vec![x as u64];
            let mut stirde_ = vec![(x_s / unit) as i64];

            shape.append(&mut shape_);
            stride.append(&mut stirde_);
        }
        2 => {
            let &[x, y] = tensor_layout.shape() else {
                unreachable!()
            };
            let &[x_s, y_s] = tensor_layout.strides() else {
                unreachable!()
            };

            get_static! {
                x y
                x_s y_s
            }

            let mut shape_ = vec![x as u64, y as u64];
            let mut stirde_ = vec![(x_s / unit) as i64, (y_s / unit) as i64];

            shape.append(&mut shape_);
            stride.append(&mut stirde_);
        }
        3 => {
            let &[x, y, z] = tensor_layout.shape() else {
                unreachable!()
            };
            let &[x_s, y_s, z_s] = tensor_layout.strides() else {
                unreachable!()
            };

            get_static! {
                x y z
                x_s y_s z_s
            }

            let mut shape_ = vec![x as u64, y as u64, z as u64];
            let mut stirde_ = vec![
                (x_s / unit) as i64,
                (y_s / unit) as i64,
                (z_s / unit) as i64,
            ];

            shape.append(&mut shape_);
            stride.append(&mut stirde_);
        }
        _ => (),
    }
    Ok((shape, stride))
}

pub fn create_tensor_decriptor(
    shape: &mut Vec<u64>,
    stride: &mut Vec<i64>,
    ty: DigitLayout,
) -> *mut *mut TensorDescriptor {
    let desc: infiniopTensorDescriptor_t = std::ptr::null_mut();
    let desc_ptr = Box::into_raw(Box::new(desc)) as *mut _;
    let ndim = shape.len();
    if stride.is_empty() {
        let mut stride_ = calculate_strides(shape);
        stride.append(&mut stride_);
    }

    match ty {
        F16 => {
            let dt_num = 84542721u32;
            let dt: DataLayout = unsafe { std::mem::transmute(dt_num) };
            infiniops!(infiniopCreateTensorDescriptor(
                desc_ptr,
                ndim as u64,
                shape.as_mut_ptr(),
                stride.as_mut_ptr(),
                dt
            ));
        }
        F32 => {
            let dt_num = 135727361u32;
            let dt: DataLayout = unsafe { std::mem::transmute(dt_num) };
            infiniops!(infiniopCreateTensorDescriptor(
                desc_ptr,
                ndim as u64,
                shape.as_mut_ptr(),
                stride.as_mut_ptr(),
                dt
            ));
        }
        U32 => {
            let dt_num = 2099201u32;
            let dt: DataLayout = unsafe { std::mem::transmute(dt_num) };
            infiniops!(infiniopCreateTensorDescriptor(
                desc_ptr,
                ndim as u64,
                shape.as_mut_ptr(),
                stride.as_mut_ptr(),
                dt
            ));
        }
        U64 => {
            let dt_num = 4198401u32;
            let dt: DataLayout = unsafe { std::mem::transmute(dt_num) };
            infiniops!(infiniopCreateTensorDescriptor(
                desc_ptr,
                ndim as u64,
                shape.as_mut_ptr(),
                stride.as_mut_ptr(),
                dt
            ));
        }
        _ => {}
    }
    desc_ptr
}

pub fn create_tensor_decriptor_from_tensorlayout(
    tensor_layout: &TensorLayout,
) -> *mut *mut TensorDescriptor {
    let (mut a_shape, mut a_stride) = get_shape_stride(tensor_layout).unwrap();
    create_tensor_decriptor(&mut a_shape, &mut a_stride, tensor_layout.dt())
}

fn calculate_strides(shape: &[u64]) -> Vec<i64> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut current_stride = 1;

    for &dim_size in shape.iter().rev() {
        strides.push(current_stride as i64);
        current_stride *= dim_size;
    }

    strides.reverse();
    strides
}
