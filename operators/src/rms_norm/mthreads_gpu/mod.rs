// include!(concat!(env!("OUT_DIR"), "/opbindings.rs"));
use super::{Args, RmsNorm};
use crate::handle::mthreads_gpu::opbindings::infiniopRMSNormDescriptor_t;
use crate::handle::mthreads_gpu::tensor::create_tensor_decriptor_from_tensorlayout;
use crate::{
    infiniops,
    mthreads_gpu::{Handle, Musa},
    ByteOf, LaunchError, QueueAlloc, SchemeError, Workspace,
};
use mudrv::AsRaw;
use std::ffi::c_void;
use std::sync::Arc;

pub struct Operator {
    handle: Arc<Handle>,
}

impl RmsNorm<Musa> for Operator {}

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
        let Args {
            y_layout,
            y_base,
            x_layout,
            x_base,
            w_layout,
            w_base,
            epsilon,
        } = args;

        let x = x_base.cast::<c_void>();
        let y = y_base.cast::<c_void>();
        let w = w_base.cast::<c_void>();
        let handle = self.handle.get_handle();

        let y_desc_ptr = create_tensor_decriptor_from_tensorlayout(y_layout);
        let x_desc_ptr = create_tensor_decriptor_from_tensorlayout(x_layout);
        let w_desc_ptr = create_tensor_decriptor_from_tensorlayout(w_layout);
        let mut op_desc: infiniopRMSNormDescriptor_t = std::ptr::null_mut();

        infiniops!(infiniopCreateRMSNormDescriptor(
            handle,
            &mut op_desc,
            *y_desc_ptr,
            *x_desc_ptr,
            *w_desc_ptr,
            *epsilon
        ));
        let mut worksize: u64 = 0;
        infiniops!(infiniopGetRMSNormWorkspaceSize(op_desc, &mut worksize));
        let mut workspace = Workspace::new(queue_alloc, workspace, worksize as usize);
        infiniops!(infiniopRMSNorm(
            op_desc,
            workspace.as_mut_ptr().cast::<c_void>(),
            worksize,
            y,
            x as *mut c_void,
            w as *mut c_void,
            queue_alloc.queue().as_raw().cast()
        ));
        infiniops!(infiniopDestroyRMSNormDescriptor(op_desc));
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Operator};
    use crate::{mthreads_gpu::Musa, Hardware, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F64},
        DigitLayout,
    };

    fn dyn_args<H: Hardware>(dt_w: DigitLayout, dt_a: DigitLayout, d: usize) -> Args<H> {
        use crate::dyn_;
        use std::ptr::{null, null_mut};
        Args {
            y_layout: TensorLayout::new_dyn(dt_a, &[dyn_(), d.into()], &[dyn_(); 2]),
            y_base: null_mut(),
            x_layout: TensorLayout::new_dyn(dt_a, &[dyn_(), d.into()], &[dyn_(); 2]),
            x_base: null(),
            w_layout: TensorLayout::new_dyn(dt_w, &[d.into()], &[dyn_()]),
            w_base: null(),
            epsilon: 1e-5,
        }
    }

    fn args<H: Hardware>(
        dt_w: DigitLayout,
        dt_a: DigitLayout,
        n: usize,
        d: usize,
        y_base: *mut H::Byte,
        x_base: *const H::Byte,
        w_base: *const H::Byte,
    ) -> Args<H> {
        let layout = TensorLayout::new_contiguous(dt_a, &[n, d]);
        Args {
            y_layout: layout.clone(),
            y_base,
            x_layout: layout,
            x_base,
            w_layout: TensorLayout::new_contiguous(dt_w, &[d]),
            w_base,
            epsilon: 1e-5,
        }
    }

    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::{
            common_cpu::{Cpu, ThisThread},
            mthreads_gpu::cast_load,
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

        // RMS_Norm 在 1024维以下的时候会报错
        for k in 10..=13 {
            let n: usize = 4;
            let d = 1 << k;
            cpu_op.scheme(&dyn_args(F64, F64, d), 0).unwrap();
            musa_op.scheme(&dyn_args(F16, F16, d), 0).unwrap();

            let mut x = vec![0.0f64; n * d];
            let mut w = vec![0.0f64; d];
            rand::thread_rng().fill(&mut x[..]);
            rand::thread_rng().fill(&mut w[..]);

            let y_ans = musa.apply(|ctx| {
                let stream = ctx.stream();
                let mut y = stream.malloc::<f16>(n * d);
                let x = cast_load(&x, f16::from_f64, &stream);
                let w = cast_load(&w, f16::from_f64, &stream);

                musa_op
                    .launch(
                        &args(
                            F16,
                            F16,
                            n,
                            d,
                            y.as_mut_ptr().cast(),
                            x.as_ptr().cast(),
                            w.as_ptr().cast(),
                        ),
                        &mut [],
                        &stream,
                    )
                    .unwrap();
                let mut host = vec![f16::ZERO; n * d];
                memcpy_d2h(&mut host, &y);
                host
            });

            let mut y_ref = vec![0.0; n * d];
            cpu_op
                .launch(
                    &args(
                        F64,
                        F64,
                        n,
                        d,
                        y_ref.as_mut_ptr().cast(),
                        x.as_ptr().cast(),
                        w.as_ptr().cast(),
                    ),
                    &mut [],
                    &ThisThread,
                )
                .unwrap();

            let diff = y_ref
                .into_par_iter()
                .zip(y_ans)
                .map(|(a, b)| Diff::new(a, b.to_f64()))
                .collect::<Vec<_>>();

            let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 1e-3);
            diff.into_iter().for_each(|diff| ec.push(diff));
            println!("{ec}");

            let (out, count) = ec.summary();
            assert!(out * 1000 <= count);
        }
    }
}
