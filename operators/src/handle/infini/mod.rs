use crate::{Alloc, Hardware, QueueAlloc, QueueOf};
use infini_rt::{
    DevBlob, DevByte, Stream, DEVICE_ASCEND, DEVICE_CAMBRICON, DEVICE_CPU, DEVICE_NVIDIA,
};
use std::{ops::Deref, sync::Arc};

pub struct Device {
    device: infini_rt::Device,
    handle: Arc<infini_op::Handle>,
}

impl Device {
    #[inline]
    pub fn cpu() -> Self {
        Self::new(DEVICE_CPU, 0)
    }

    #[inline]
    pub fn nv_gpu(id: usize) -> Self {
        Self::new(DEVICE_NVIDIA, id)
    }

    #[inline]
    pub fn cambricon_mlu(id: usize) -> Self {
        Self::new(DEVICE_CAMBRICON, id)
    }

    #[inline]
    pub fn ascend_npu(id: usize) -> Self {
        Self::new(DEVICE_ASCEND, id)
    }

    fn new(ty: infini_rt::DeviceType, id: usize) -> Self {
        use infini_op::bindings::Device::*;
        Self {
            device: infini_rt::Device { ty, id: id as _ },
            handle: Arc::new(infini_op::Handle::new(
                match ty {
                    DEVICE_CPU => DevCpu,
                    DEVICE_NVIDIA => DevNvGpu,
                    DEVICE_CAMBRICON => DevCambriconMlu,
                    DEVICE_ASCEND => DevAscendNpu,
                    _ => unreachable!("unknown device type"),
                },
                id as _,
            )),
        }
    }

    #[inline]
    pub(crate) fn handle(&self) -> &Arc<infini_op::Handle> {
        &self.handle
    }
}

impl Deref for Device {
    type Target = infini_rt::Device;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Hardware for Device {
    type Byte = DevByte;
    type Queue<'ctx> = Stream;
}

impl Alloc<DevBlob> for Device {
    #[inline]
    fn alloc(&self, size: usize) -> DevBlob {
        self.device.malloc::<u8>(size)
    }

    #[inline]
    fn free(&self, _mem: DevBlob) {}
}

impl Alloc<DevBlob> for Stream {
    #[inline]
    fn alloc(&self, size: usize) -> DevBlob {
        self.malloc::<u8>(size)
    }

    #[inline]
    fn free(&self, mem: DevBlob) {
        self.free(mem)
    }
}

impl QueueAlloc for Stream {
    type Hardware = Device;
    type DevMem = DevBlob;
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Hardware> {
        self
    }
}
