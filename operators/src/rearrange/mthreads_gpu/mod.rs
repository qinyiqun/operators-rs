use super::{args::Scheme, Args, Rearrange};
use crate::handle::mthreads_gpu::opbindings::infiniopRearrangeDescriptor_t;
use crate::handle::mthreads_gpu::tensor::create_tensor_decriptor_from_tensorlayout;
use crate::mthreads_gpu::{Handle, Musa};
use crate::{infiniops, ByteOf, LaunchError, QueueAlloc, SchemeError};
use mudrv::memcpy_d2d;
use mudrv::AsRaw;
use std::ffi::c_void;
use std::{
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::Arc,
};

pub struct Operator {
    handle: Arc<Handle>,
}

impl Rearrange<Musa> for Operator {}

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
        _workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let scheme = Scheme::new(args)?;
        let unit = scheme.unit();
        if scheme.ndim() == 0 {
            let dst = unsafe { from_raw_parts_mut(args.dst_base, unit) };
            let src = unsafe { from_raw_parts(args.src_base, unit) };
            memcpy_d2d(dst, src);
            return Ok(());
        }

        let handle = self.handle.get_handle();
        let src_desc_ptr = create_tensor_decriptor_from_tensorlayout(&args.src_layout);
        let dst_desc_ptr = create_tensor_decriptor_from_tensorlayout(&args.dst_layout);

        let dst = args.dst_base.cast::<c_void>();
        let src = args.src_base.cast::<c_void>();
        let mut op_desc: infiniopRearrangeDescriptor_t = std::ptr::null_mut();

        infiniops!(infiniopCreateRearrangeDescriptor(
            handle,
            &mut op_desc,
            *dst_desc_ptr,
            *src_desc_ptr
        ));
        infiniops!(infiniopRearrange(
            op_desc,
            dst,
            src as *mut c_void,
            queue_alloc.queue().as_raw().cast()
        ));
        infiniops!(infiniopDestroyRearrangeDescriptor(op_desc));
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Musa, Operator};
    use crate::{ConstPtr, Hardware, MutPtr, Operator as _, TensorLayout};
    use digit_layout::{types as ty, DigitLayout};

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use crate::dyn_;
        use std::ptr::{null, null_mut};
        Args {
            dst_layout: TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]),
            dst_base: null_mut(),
            src_layout: TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]),
            src_base: null(),
        }
    }

    fn args<H: Hardware>(
        dt: DigitLayout,
        shape: &[usize],
        s_src: &[isize],
        s_dst: &[isize],
        src_base: ConstPtr<H>,
        dst_base: MutPtr<H>,
    ) -> Args<H> {
        Args {
            dst_layout: TensorLayout::new(dt, shape, s_dst),
            dst_base,
            src_layout: TensorLayout::new(dt, shape, s_src),
            src_base,
        }
    }

    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::common_cpu::{Cpu, ThisThread};
        use mudrv::memcpy_d2h;
        use ndarray_layout::{ArrayLayout, Endian::BigEndian};
        use rand::Rng;

        let Some(musa) = Musa::init() else {
            return;
        };

        let dt = ty::U32;

        let mut cpu_op = RefOp::new(&Cpu);
        let mut musa_op = Operator::new(&musa);
        cpu_op.scheme(&dyn_args(dt), 0).unwrap();
        musa_op.scheme(&dyn_args(dt), 0).unwrap();

        let nh = 32;
        let seq = 7;
        let dh = 128;
        let mut src = vec![0u32; nh * seq * dh];
        rand::thread_rng().fill(&mut src[..]);

        let s_src = ArrayLayout::<3>::new_contiguous(&[nh, seq, dh], BigEndian, dt.nbytes());
        let s_dst = ArrayLayout::<3>::new_contiguous(&[seq, nh, dh], BigEndian, dt.nbytes())
            .transpose(&[1, 0]);

        let dst_ans = musa.apply(|ctx| {
            let stream = ctx.stream();
            let src = stream.from_host(&src);
            let mut dst = stream.malloc::<u8>(src.len());
            musa_op
                .launch(
                    &args(
                        dt,
                        &[nh, seq, dh],
                        s_src.strides(),
                        s_dst.strides(),
                        src.as_ptr().cast(),
                        dst.as_mut_ptr().cast(),
                    ),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![0u32; nh * seq * dh];
            memcpy_d2h(&mut host, &dst);
            host
        });

        let mut dst_ref = vec![0u32; seq * nh * dh];
        cpu_op
            .launch(
                &args(
                    dt,
                    &[nh, seq, dh],
                    s_src.strides(),
                    s_dst.strides(),
                    src.as_ptr().cast(),
                    dst_ref.as_mut_ptr().cast(),
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();
        assert_eq!(dst_ans, dst_ref);
    }
}
