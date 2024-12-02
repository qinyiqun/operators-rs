use super::{args::Meta, fill_pos, Args, Rope, Seq, SinCosTable};
use crate::{
    get_static, infiniops,
    mthreads_gpu::{Handle, Musa},
    type_not_support, Blob, LaunchError, QueueAlloc, SchemeError, Workspace,
};
use digit_layout::{types as ty, DigitLayout};

use crate::handle::mthreads_gpu::opbindings::infiniopRoPEDescriptor_t;
use crate::handle::mthreads_gpu::tensor::{
    create_tensor_decriptor, create_tensor_decriptor_from_tensorlayout,
};
use mudrv::AsRaw;
use std::{ffi::c_void, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
}

impl Rope<Musa> for Operator {
    fn build_sincos<QA>(
        _dt: DigitLayout,
        nctx: usize,
        dh: usize,
        queue_alloc: &QA,
    ) -> SinCosTable<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let sin_cos_table = generate_sin_cos_tables(2 * nctx, dh, &1e4);
        let mut table = queue_alloc.alloc(sin_cos_table.len() * 4);
        queue_alloc.queue().memcpy_h2d(&mut table, &sin_cos_table);
        SinCosTable { nctx, mem: table }
    }

    fn build_pos<I, QA>(
        dt: digit_layout::DigitLayout,
        nt: usize,
        iter: I,
        queue_alloc: &QA,
    ) -> QA::DevMem
    where
        I: IntoIterator<Item = Seq>,
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let mut host = Blob::new(dt.nbytes() * nt);
        match dt {
            ty::U32 => fill_pos(host.as_mut_ptr().cast::<u32>(), nt, iter),
            ty::U64 => fill_pos(host.as_mut_ptr().cast::<u64>(), nt, iter),
            _ => todo!(),
        }

        let mut blob = queue_alloc.alloc(host.len());
        queue_alloc.queue().memcpy_h2d(&mut blob, &host);
        blob
    }
}

impl crate::Operator for Operator {
    type Hardware = Musa;
    type TopoNode = Musa;
    type Args = Args<Musa>;

    fn new(node: &Self::TopoNode) -> Self {
        Self {
            handle: node.0.clone(),
        }
    }

    fn scheme(
        &mut self,
        args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        let Meta { dt_t, dt_p, .. } = args.meta()?;
        if dt_t == digit_layout::types::F16 || dt_p == digit_layout::types::F32 {
            Ok(0)
        } else {
            Err(type_not_support(""))
        }
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        workspace: &mut [crate::ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: crate::QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta { dt_t, dt_p, .. } = args.meta()?;

        if dt_t != digit_layout::types::F16 || dt_p != digit_layout::types::U64 {
            return Err(type_not_support("").into());
        }

        let Args {
            t_layout,
            t_base,
            p_layout,
            p_base,
            sin_layout,
            ..
        } = args;

        let &[_, _, dh] = t_layout.shape() else {
            unreachable!()
        };

        let &[nctx, _] = sin_layout.shape() else {
            unreachable!()
        };

        get_static! {
            nctx dh
        }

        let t = t_base.cast::<c_void>();
        let p = p_base.cast::<c_void>();

        let mut sin_shape = vec![2 * nctx as u64, dh as u64];
        let mut cos_shape = vec![2 * nctx as u64, dh as u64];
        let mut sin_stride = vec![];
        let mut cos_stride = vec![];

        let handle = self.handle.get_handle();

        let t_desc_ptr = create_tensor_decriptor_from_tensorlayout(t_layout);
        let p_desc_ptr = create_tensor_decriptor_from_tensorlayout(p_layout);
        let sin_desc_ptr =
            create_tensor_decriptor(&mut sin_shape, &mut sin_stride, digit_layout::types::F32);
        let cos_desc_ptr =
            create_tensor_decriptor(&mut cos_shape, &mut cos_stride, digit_layout::types::F32);
        let mut op_desc: infiniopRoPEDescriptor_t = std::ptr::null_mut();

        infiniops!(infiniopCreateRoPEDescriptor(
            handle,
            &mut op_desc,
            *t_desc_ptr,
            *p_desc_ptr,
            *sin_desc_ptr,
            *cos_desc_ptr
        ));
        let mut worksize: u64 = 0;
        infiniops!(infiniopGetRoPEWorkspaceSize(op_desc, &mut worksize));
        let mut workspace = Workspace::new(queue_alloc, workspace, worksize as usize);
        infiniops!(infiniopRoPE(
            op_desc,
            workspace.as_mut_ptr().cast::<c_void>(),
            worksize,
            t,
            p,
            args.sin_base.cast::<c_void>(),
            args.cos_base.cast::<c_void>(),
            queue_alloc.queue().as_raw().cast()
        ));
        infiniops!(infiniopDestroyRoPEDescriptor(op_desc));
        queue_alloc.queue().synchronize();
        Ok(())
    }
}

pub fn generate_sin_cos_tables(max_seq_len: usize, dim: usize, theta: &f32) -> Vec<f32> {
    let mut sin_cos_table = vec![0.0f32; dim * max_seq_len * 2];

    let half_dh = dim / 2;
    for i in 0..max_seq_len {
        for j in 0..half_dh {
            let _sin = (i as f32 / theta.powf(j as f32 / half_dh as f32)).sin();
            let _cos = (i as f32 / theta.powf(j as f32 / half_dh as f32)).cos();
            sin_cos_table[i * dim + 2 * j] = _sin;
            sin_cos_table[i * dim + 2 * j + 1] = _sin;
            sin_cos_table[dim * max_seq_len + i * dim + 2 * j] = _cos;
            sin_cos_table[dim * max_seq_len + i * dim + 2 * j + 1] = _cos;
        }
    }
    sin_cos_table
}
