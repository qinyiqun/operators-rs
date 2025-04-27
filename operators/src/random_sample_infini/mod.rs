use crate::{get_static, infini::Device, ByteOf, LaunchError, QueueAlloc, SchemeError, Workspace};

mod args;
pub use args::{Args, SampleArgs};
use infini_op::{infiniop, AsRaw, Descriptor};

crate::op_trait!(RandomSampleInfini);

pub struct Operator(Device);

impl RandomSampleInfini<Device> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Device;
    type TopoNode = Device;
    type Args = Args<Device>;

    fn new(node: &Self::TopoNode) -> Self {
        Self(node.clone())
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
            result_layout,
            result_base,
            probs_layout,
            probs_base,
            config,
            seed,
        } = args;

        let dt_r = result_layout.dt();
        let dt_p = probs_layout.dt();

        let &[n] = probs_layout.shape() else {
            unreachable!()
        };

        get_static! {
            n
        }

        let result = infini_op::Tensor::new(dt_r, [1], [1]);
        let probs = infini_op::Tensor::new(dt_p, [n], [1]);

        let descriptor = Descriptor::new(
            |ptr| {
                infiniop!(infiniopCreateRandomSampleDescriptor(
                    self.0.handle().as_raw(),
                    ptr,
                    result.as_raw(),
                    probs.as_raw(),
                ))
            },
            infini_op::bindings::infiniopDestroyRandomSampleDescriptor,
        );

        let mut workspace_size = 0;
        infiniop!(infiniopGetRandomSampleWorkspaceSize(
            descriptor.as_raw(),
            &mut workspace_size
        ));
        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size as _);
        infiniop!(infiniopRandomSample(
            descriptor.as_raw(),
            workspace.as_mut_ptr().cast(),
            workspace_size,
            result_base.cast(),
            probs_base.cast(),
            *seed,
            config.top_p,
            config.top_k as i32,
            config.temperature,
            queue_alloc.queue().as_void_ptr()
        ));

        Ok(())
    }
}

