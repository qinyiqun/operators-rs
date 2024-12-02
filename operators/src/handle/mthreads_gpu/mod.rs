#![cfg(use_musa)]
#[macro_use]
#[allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]

pub mod opbindings {
    include!(concat!(env!("OUT_DIR"), "/opbindings.rs"));

    #[macro_export]
    macro_rules! infiniops {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::handle::mthreads_gpu::opbindings::*;
            #[allow(unused_unsafe, clippy::macro_metavars_in_unsafe)]
            let error = unsafe { $f };
            assert_eq!(error, infiniopStatus_t::STATUS_SUCCESS);
        }};
    }
}
use std::{ptr::null_mut, sync::Arc};

use crate::handle::mthreads_gpu::opbindings::{infiniopHandle_t, Device};
use crate::{Alloc, Hardware, QueueAlloc, QueueOf};
use mudrv::{Context, CurrentCtx, DevMem, Stream};

pub mod tensor;

pub struct Musa(pub(crate) Arc<Handle>);

pub(crate) struct Handle {
    context: Context,
    handle: infiniopHandle_t,
}

impl Handle {
    pub(crate) fn get_handle(&self) -> infiniopHandle_t {
        self.handle
    }
}

impl Hardware for Musa {
    type Byte = mudrv::DevByte;
    type Queue<'ctx> = mudrv::Stream<'ctx>;
}

impl<'ctx> Alloc<DevMem<'ctx>> for &'ctx CurrentCtx {
    #[inline]
    fn alloc(&self, size: usize) -> DevMem<'ctx> {
        self.malloc::<u8>(size)
    }

    #[inline]
    fn free(&self, _mem: DevMem<'ctx>) {}
}

impl<'ctx> Alloc<DevMem<'ctx>> for Stream<'ctx> {
    #[inline]
    fn alloc(&self, size: usize) -> DevMem<'ctx> {
        self.malloc::<u8>(size)
    }

    #[inline]
    fn free(&self, _mem: DevMem<'ctx>) {
        // mem.drop_on(self)
    }
}

impl<'ctx> QueueAlloc for Stream<'ctx> {
    type Hardware = Musa;
    type DevMem = DevMem<'ctx>;
    fn queue(&self) -> &QueueOf<Self::Hardware> {
        self
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub low_diversity_cache: usize,
    pub medium_diversity_cache: usize,
    pub high_diversity_cache: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            low_diversity_cache: 4,
            medium_diversity_cache: 16,
            high_diversity_cache: 64,
        }
    }
}

impl Musa {
    #[inline]
    pub fn new(context: Context) -> Self {
        let device = Device::DevMtGpu;
        let mut handle: infiniopHandle_t = null_mut();
        infiniops!(infiniopCreateHandle(&mut handle, device, 0));
        Self(Arc::new(Handle { context, handle }))
    }

    #[inline]
    pub fn apply<T>(&self, f: impl FnOnce(&CurrentCtx) -> T) -> T {
        self.0.context.apply(f)
    }

    #[cfg(test)]
    pub(crate) fn init() -> Option<Self> {
        if let Err(mudrv::NoDevice) = mudrv::init() {
            return None;
        }
        Some(Self::new(mudrv::Device::new(0).retain_primary()))
    }
}

#[cfg(test)]
pub(crate) fn cast_load<'ctx, T, U, F>(
    val: &[T],
    f: F,
    stream: &Stream<'ctx>,
) -> mudrv::DevMem<'ctx>
where
    T: Sync + Copy,
    U: Send + Copy,
    F: Sync + Fn(T) -> U,
{
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
    let mut host = stream.ctx().malloc_host::<U>(val.len());
    let host = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast(), val.len()) };
    host.into_par_iter().zip(val).for_each(|(y, x)| *y = f(*x));
    stream.from_host(host)
}

#[test]
fn test_create_handle() {
    use opbindings;
    let device = opbindings::Device::DevMtGpu;
    let handle = opbindings::HandleStruct { device };
    let handle_ptr = &mut Box::into_raw(Box::new(handle)) as *mut _;
    infiniops!(opbindings::infiniopCreateHandle(handle_ptr, device, 3));
}
