use super::{args::Meta, Args, FusedSoftmax};
use crate::handle::mthreads_gpu::opbindings::infiniopCausalSoftmaxDescriptor_t;
use crate::handle::mthreads_gpu::tensor::create_tensor_decriptor_from_tensorlayout;
use crate::mthreads_gpu::{Handle, Musa};
use crate::{infiniops, type_not_support, ByteOf, LaunchError, QueueAlloc, SchemeError, Workspace};
use mudrv::AsRaw;
use std::ffi::c_void;
use std::sync::Arc;

pub struct Operator {
    handle: Arc<Handle>,
}

impl FusedSoftmax<Musa> for Operator {}

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
        let Meta { dt } = args.meta()?;
        if dt == digit_layout::types::F16 {
            Ok(0)
        } else {
            Err(type_not_support(""))
        }
    }

    fn launch<T>(
        &self,
        args: &Self::Args,
        workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &T,
    ) -> Result<(), LaunchError>
    where
        T: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta { dt } = args.meta()?;
        let Args {
            att_layout,
            att_base,
        } = args;

        if dt != digit_layout::types::F16 {
            return Err(type_not_support("").into());
        }

        let a = att_base.cast::<c_void>();
        let handle = self.handle.get_handle();

        let a_desc_ptr = create_tensor_decriptor_from_tensorlayout(att_layout);
        let mut op_desc: infiniopCausalSoftmaxDescriptor_t = std::ptr::null_mut();

        infiniops!(infiniopCreateCausalSoftmaxDescriptor(
            handle,
            &mut op_desc,
            *a_desc_ptr
        ));
        let mut worksize: u64 = 0;
        infiniops!(infiniopGetCausalSoftmaxWorkspaceSize(
            op_desc,
            &mut worksize
        ));
        let mut workspace = Workspace::new(queue_alloc, workspace, worksize as usize);
        infiniops!(infiniopCausalSoftmax(
            op_desc,
            workspace.as_mut_ptr().cast::<c_void>(),
            worksize,
            a,
            queue_alloc.queue().as_raw().cast()
        ));
        infiniops!(infiniopDestroyCausalSoftmaxDescriptor(op_desc));
        queue_alloc.queue().synchronize();
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Operator};
    use crate::{Hardware, Operator as _, TensorLayout};
    use digit_layout::{types as ty, DigitLayout};

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use crate::dyn_;
        use std::ptr::null_mut;
        Args {
            att_layout: TensorLayout::new_dyn(dt, &[dyn_(); 3], &[dyn_(); 3]),
            att_base: null_mut(),
        }
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        nh: usize,
        seq_len: usize,
        att_len: usize,
        att_base: *mut H::Byte,
    ) -> Args<H> {
        Args {
            att_layout: TensorLayout::new_contiguous(dt, &[nh, seq_len, att_len]),
            att_base,
        }
    }

    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::{
            common_cpu::{Cpu, ThisThread},
            mthreads_gpu::{cast_load, Musa},
            test_utils::{Diff, ErrorCollector},
        };

        use half::f16;
        use mudrv::memcpy_d2h;
        use rand::Rng;
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

        let Some(musa) = Musa::init() else {
            return;
        };

        let mut cpu_op = RefOp::new(&Cpu);
        let mut musa_op = Operator::new(&musa);
        cpu_op.scheme(&dyn_args(ty::F64), 0).unwrap();
        musa_op.scheme(&dyn_args(ty::F16), 0).unwrap();

        let nh = 32;
        for (seq_len, att_len) in [(1, 511), (1, 2048), (7, 511), (7, 2048)] {
            let mut att = vec![0.0f64; nh * seq_len * att_len];
            rand::thread_rng().fill(&mut att[..]);

            let att_ans = musa.apply(|ctx| {
                let stream = ctx.stream();
                let mut att = cast_load(&att, f16::from_f64, &stream);
                musa_op
                    .launch(
                        &args(ty::F16, nh, seq_len, att_len, att.as_mut_ptr().cast()),
                        &mut [],
                        &stream,
                    )
                    .unwrap();
                let mut host = vec![f16::ZERO; nh * seq_len * att_len];
                memcpy_d2h(&mut host, &att);
                host
            });
            let mut att_ref = att;
            cpu_op
                .launch(
                    &args(ty::F64, nh, seq_len, att_len, att_ref.as_mut_ptr().cast()),
                    &mut [],
                    &ThisThread,
                )
                .unwrap();

            let diff = att_ref
                .into_par_iter()
                .zip(att_ans.clone())
                .map(|(a, b)| Diff::new(a, b.to_f64()))
                .collect::<Vec<_>>();

            let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
            diff.into_iter().for_each(|diff| ec.push(diff));
            println!("{ec}");

            let (out, count) = ec.summary();
            assert!(out * 1000 <= count);
        }
    }
}
