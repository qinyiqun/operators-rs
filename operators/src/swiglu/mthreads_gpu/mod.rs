use super::{args::Meta, Args, Swiglu};
use crate::handle::mthreads_gpu::tensor::create_tensor_decriptor_from_tensorlayout;
use crate::{
    infiniops,
    mthreads_gpu::{opbindings::infiniopSwiGLUDescriptor_t, Handle, Musa},
};
use crate::{type_not_support, LaunchError, SchemeError};
use mudrv::AsRaw;
use std::ffi::c_void;
use std::sync::Arc;

pub struct Operator {
    handle: Arc<Handle>,
}

impl Swiglu<Musa> for Operator {}

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
        let Meta { dt, .. } = args.meta()?;
        if dt == digit_layout::types::F16 {
            Ok(0)
        } else {
            Err(type_not_support(""))
        }
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        _workspace: &mut [crate::ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: crate::QueueAlloc<Hardware = Self::Hardware>,
    {
        let Args {
            gate_layout,
            gate_base,
            up_layout,
            up_base,
        } = args;

        let a = up_base.cast::<c_void>();
        let b = gate_base.cast::<c_void>();
        let handle = self.handle.get_handle();
        let a_desc_ptr = create_tensor_decriptor_from_tensorlayout(up_layout);
        let b_desc_ptr = create_tensor_decriptor_from_tensorlayout(gate_layout);
        let mut op_desc: infiniopSwiGLUDescriptor_t = std::ptr::null_mut();

        infiniops!(infiniopCreateSwiGLUDescriptor(
            handle,
            &mut op_desc,
            *a_desc_ptr,
            *a_desc_ptr,
            *b_desc_ptr
        ));
        infiniops!(infiniopSwiGLU(
            op_desc,
            b,
            a,
            b,
            queue_alloc.queue().as_raw() as *mut c_void
        ));
        infiniops!(infiniopDestroySwiGLUDescriptor(op_desc));
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Operator};
    use crate::{dyn_, Hardware, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F64},
        DigitLayout,
    };

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use std::ptr::{null, null_mut};
        let layout = TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]);
        Args {
            gate_layout: layout.clone(),
            gate_base: null_mut(),
            up_layout: layout,
            up_base: null(),
        }
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        n: usize,
        d: usize,
        gate_base: *mut H::Byte,
        up_base: *const H::Byte,
    ) -> Args<H> {
        let layout = TensorLayout::new_contiguous(dt, &[n, d]);
        Args {
            gate_layout: layout.clone(),
            gate_base,
            up_layout: layout,
            up_base,
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
        let mut musa_op = Operator::new(&musa);
        musa_op.scheme(&dyn_args(F16), 0).unwrap();

        let mut cpu_op = RefOp::new(&Cpu);
        cpu_op.scheme(&dyn_args(F64), 0).unwrap();

        let n = 8;
        let d = 16;

        let mut gate = vec![0.0f64; n * d];
        let mut up = vec![0.0f64; n * d];
        rand::thread_rng().fill(&mut gate[..]);
        rand::thread_rng().fill(&mut up[..]);
        let gate = gate;
        let up = up;

        let gate_ans = musa.apply(|ctx| {
            let stream = ctx.stream();
            let mut gate = cast_load(&gate, f16::from_f64, &stream);
            let up = cast_load(&up, f16::from_f64, &stream);

            musa_op
                .launch(
                    &args(F16, n, d, gate.as_mut_ptr().cast(), up.as_ptr().cast()),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![f16::ZERO; n * d];
            memcpy_d2h(&mut host, &gate);
            host
        });

        let mut gate_ref = gate;
        cpu_op
            .launch(
                &args(F64, n, d, gate_ref.as_mut_ptr().cast(), up.as_ptr().cast()),
                &mut [],
                &ThisThread,
            )
            .unwrap();

        let diff = gate_ref
            .into_par_iter()
            .zip(gate_ans)
            .map(|(a, b)| Diff::new(a, b.to_f64()))
            .collect::<Vec<_>>();

        let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
        diff.into_iter().for_each(|diff| ec.push(diff));
        println!("{ec}");

        let (out, count) = ec.summary();
        assert!(out * 1000 <= count);
    }
}
