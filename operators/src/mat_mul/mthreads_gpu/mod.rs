use super::{Args, MatMul};
use crate::handle::mthreads_gpu::opbindings::infiniopMatmulDescriptor_t;
use crate::handle::mthreads_gpu::tensor::create_tensor_decriptor_from_tensorlayout;
use crate::{
    infiniops,
    mthreads_gpu::{Handle, Musa},
    ByteOf, LaunchError, QueueAlloc, SchemeError, Workspace,
};
use mudrv::AsRaw;
use std::{ffi::c_void, sync::Arc};

pub struct Operator {
    handle: Arc<Handle>,
}

impl MatMul<Musa> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Musa;
    type TopoNode = Musa;
    type Args = Args<Musa>;

    #[inline]
    fn new(node: &Self::TopoNode) -> Self {
        Self {
            handle: node.0.clone(),
        }
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        Ok(0)
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let &Args {
            c_base,
            beta,
            a_base,
            b_base,
            alpha,
            ..
        } = args;

        let c = c_base.cast::<c_void>();
        let a = a_base.cast::<c_void>();
        let b = b_base.cast::<c_void>();
        let handle = self.handle.get_handle();

        let a_desc_ptr = create_tensor_decriptor_from_tensorlayout(&args.a_layout);
        let b_desc_ptr = create_tensor_decriptor_from_tensorlayout(&args.b_layout);
        let c_desc_ptr = create_tensor_decriptor_from_tensorlayout(&args.c_layout);
        let mut op_desc: infiniopMatmulDescriptor_t = std::ptr::null_mut();

        infiniops!(infiniopCreateMatmulDescriptor(
            handle,
            &mut op_desc,
            *c_desc_ptr,
            alpha,
            *a_desc_ptr,
            *b_desc_ptr,
            beta
        ));
        let mut worksize: u64 = 0;
        infiniops!(infiniopGetMatmulWorkspaceSize(op_desc, &mut worksize));
        let mut workspace = Workspace::new(queue_alloc, workspace, worksize as usize);
        infiniops!(infiniopMatmul(
            op_desc,
            workspace.as_mut_ptr().cast::<c_void>(),
            worksize,
            c,
            a,
            b,
            queue_alloc.queue().as_raw().cast()
        ));
        infiniops!(infiniopDestroyMatmulDescriptor(op_desc));
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::Args;
    use crate::{Hardware, TensorLayout};
    use digit_layout::DigitLayout;

    const ALPHA: f32 = 0.5;
    const BETA: f32 = 1.;

    fn args<H: Hardware>(
        dt: DigitLayout,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        c_base: *mut H::Byte,
        a_base: *const H::Byte,
        b_base: *const H::Byte,
    ) -> Args<H> {
        Args {
            c_layout: TensorLayout::new_contiguous(dt, &[batch, m, n]),
            c_base,
            beta: BETA,
            a_layout: TensorLayout::new_contiguous(dt, &[batch, m, k]),
            a_base,
            b_layout: TensorLayout::new_contiguous(dt, &[batch, k, n]),
            b_base,
            alpha: ALPHA,
        }
    }

    #[test]
    fn test_launch() {
        use super::{super::common_cpu::Operator as RefOp, Operator};
        use crate::common_cpu::{Cpu, ThisThread};
        use crate::mthreads_gpu::{cast_load, Musa};
        use crate::test_utils::{Diff, ErrorCollector};
        use crate::Operator as _;
        use digit_layout::types::F64;
        use half::f16;
        use mudrv::memcpy_d2h;
        use rand::Rng;
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

        let Some(musa) = Musa::init() else {
            return;
        };

        let cpu_op = RefOp::new(&Cpu);
        let musa_op = Operator::new(&musa);
        let batch = 4;
        let k = 2046;
        let n = 2560;
        for m in [1, 7, 64, 255, 1024] {
            let mut a = vec![0.0f64; batch * m * k];
            let mut b = vec![0.0f64; batch * k * n];
            let c = vec![0.0f64; batch * m * n];

            let mut rng = rand::thread_rng();
            rng.fill(&mut a[..]);
            rng.fill(&mut b[..]);

            let a = a;
            let b = b;

            let c_ans = musa.apply(|ctx| {
                let stream = ctx.stream();

                let mut c = cast_load(&c, f16::from_f64, &stream);
                let a = cast_load(&a, f16::from_f64, &stream);
                let b = cast_load(&b, f16::from_f64, &stream);
                musa_op
                    .launch(
                        &args(
                            digit_layout::types::F16,
                            batch,
                            m,
                            n,
                            k,
                            c.as_mut_ptr().cast(),
                            a.as_ptr().cast(),
                            b.as_ptr().cast(),
                        ),
                        &mut [],
                        &stream,
                    )
                    .unwrap();

                let mut ans = vec![f16::ZERO; batch * m * n];
                memcpy_d2h(&mut ans, &c);
                ans
            });

            let mut c_ref = c;
            cpu_op
                .launch(
                    &args(
                        F64,
                        batch,
                        m,
                        n,
                        k,
                        c_ref.as_mut_ptr().cast(),
                        a.as_ptr().cast(),
                        b.as_ptr().cast(),
                    ),
                    &mut [],
                    &ThisThread,
                )
                .unwrap();

            let diff = c_ref
                .into_par_iter()
                .zip(c_ans)
                .map(|(a, b)| Diff::new(a, b.to_f64()))
                .collect::<Vec<_>>();

            let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 5e-3);
            diff.into_iter().for_each(|diff| ec.push(diff));
            println!("{ec}");

            let (out, count) = ec.summary();
            assert!(out * 1000 <= count);
        }
    }
}
