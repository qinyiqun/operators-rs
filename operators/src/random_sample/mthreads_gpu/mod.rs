use super::{args::Meta, common_cpu::Operator as RefOp, Args, Indices, RandomSample};
use crate::{
    common_cpu::{Cpu, ThisThread},
    get_static,
    mthreads_gpu::Musa,
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use mudrv::memcpy_d2h;
use std::{ptr::null, slice::from_raw_parts};

pub struct Operator;

impl RandomSample<Musa> for Operator {
    fn build_indices<QA>(_n: usize, queue_alloc: &QA) -> Indices<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        Indices {
            n: 0,
            mem: queue_alloc.alloc(1),
        }
    }
}

impl crate::Operator for Operator {
    type Hardware = Musa;
    type TopoNode = Musa;
    type Args = Args<Musa>;

    fn new(_node: &Self::TopoNode) -> Self {
        Self
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
        _queue: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Args {
            kv_pair: _,
            kv_pair_base,
            logits: _,
            logits_base,
            indices: _,
            indices_base: _,
            config,
            seed,
        } = args;
        let Meta { dt, n } = args.meta()?;
        get_static! {
            n
        }
        let unit = dt.nbytes();
        let mut host = vec![0u8; n * unit];
        memcpy_d2h(&mut host, unsafe { from_raw_parts(*logits_base, n * unit) });

        let cpu_op = RefOp::new(&Cpu);
        cpu_op
            .launch(
                &Args {
                    kv_pair_base: kv_pair_base.cast(),
                    logits_base: host.as_ptr().cast(),
                    indices_base: null(),
                    config: *config,
                    seed: *seed,
                    ..Args::layout(dt, n)
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();
        Ok(())
    }
}

#[test]
fn test_compute() {
    use super::args::SampleArgs;
    use super::{common_cpu::Operator as RefOp, KVPair};
    use crate::{
        common_cpu::{Cpu, ThisThread},
        mthreads_gpu::Musa,
        Operator as _,
    };
    use digit_layout::types as ty;
    use rand::Rng;
    use std::ptr::null;

    let Some(musa) = Musa::init() else {
        return;
    };
    let n = 32000;

    let cpu_op = RefOp::new(&Cpu);
    let mut gpu_op = Operator::new(&musa);
    println!(
        "workspace = {}",
        gpu_op
            .scheme(&Args::layout(ty::F32, n), usize::MAX)
            .unwrap()
    );

    let mut logits = vec![0.0f32; n];
    rand::thread_rng().fill(&mut logits[..]);

    // argmax
    {
        let kv_ans = musa.apply(|ctx| {
            let stream = ctx.stream();
            let logits = stream.from_host(&logits);
            let mut kv: KVPair<f32> = KVPair::new(u32::MAX, 0.0f32);

            gpu_op
                .launch(
                    &Args {
                        kv_pair_base: (&mut kv) as *mut _ as _,
                        logits_base: logits.as_ptr().cast(),
                        ..Args::layout(ty::F32, n)
                    },
                    &mut [],
                    &stream,
                )
                .unwrap();
            kv
        });

        let mut kv_ref: KVPair<f32> = KVPair::new(u32::MAX, 0.0f32);
        cpu_op
            .launch(
                &Args {
                    kv_pair_base: (&mut kv_ref) as *mut _ as _,
                    logits_base: logits.as_ptr().cast(),
                    ..Args::layout(ty::F32, n)
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();
        assert_eq!(kv_ans.idx(), kv_ref.idx());
    }

    // sample
    {
        let config = SampleArgs {
            temperature: 0.9,
            top_p: 0.9,
            top_k: 200,
        };
        let seed = 0.75;
        let kv_ans = musa.apply(|ctx| {
            let stream = ctx.stream();

            let logits = stream.from_host(&logits);
            let indices = Operator::build_indices(n, &stream).mem;
            let mut kv = KVPair::new(0, 0.0f32);

            gpu_op
                .launch(
                    &Args {
                        kv_pair_base: (&mut kv) as *mut _ as _,
                        logits_base: logits.as_ptr().cast(),
                        indices_base: indices.as_ptr().cast(),
                        config,
                        seed,
                        ..Args::layout(ty::F32, n)
                    },
                    &mut [],
                    &stream,
                )
                .unwrap();
            kv
        });

        let mut kv_ref = KVPair::new(0, 0.0f32);
        cpu_op
            .launch(
                &Args {
                    kv_pair_base: (&mut kv_ref) as *mut _ as _,
                    logits_base: logits.as_ptr().cast(),
                    indices_base: null(),
                    config,
                    seed,
                    ..Args::layout(ty::F32, n)
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();

        assert_eq!(kv_ans.idx(), kv_ref.idx());
    }
}
