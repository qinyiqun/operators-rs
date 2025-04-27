use crate::{ConstPtr, Hardware, MutPtr, TensorLayout};
use digit_layout::{types::U64, DigitLayout};
use std::ptr::{null, null_mut};

pub struct Args<H: Hardware> {
    pub result_layout: TensorLayout,
    pub result_base: MutPtr<H>,
    pub probs_layout: TensorLayout,
    pub probs_base: ConstPtr<H>,
    pub config: SampleArgs,
    pub seed: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SampleArgs {
    pub(super) temperature: f32,
    pub(super) top_p: f32,
    pub(super) top_k: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleArgsError {
    NegativeTemperature,
    NonPositiveTop,
}

impl<H: Hardware> Args<H> {
    pub fn layout(dt: DigitLayout, n: usize) -> Self {
        Args {
            result_layout: TensorLayout::new(U64, &[1], &[1]),
            result_base: null_mut(),
            probs_layout: TensorLayout::new(dt, &[n], &[dt.nbytes() as _]),
            probs_base: null(),
            config: SampleArgs {
                temperature: 0.0,
                top_p: 0.0,
                top_k: usize::MAX,
            },
            seed: 0.0,
        }
    }
}

impl Default for SampleArgs {
    #[inline]
    fn default() -> Self {
        Self::ARG_MAX
    }
}

impl SampleArgs {
    pub const ARG_MAX: Self = Self {
        temperature: 0.,
        top_p: 1.,
        top_k: usize::MAX,
    };

    pub fn new(temperature: f32, top_p: f32, top_k: usize) -> Result<Self, SampleArgsError> {
        if temperature < 0.0 {
            return Err(SampleArgsError::NegativeTemperature);
        }
        if top_k == 0 || top_p <= 0.0 {
            return Err(SampleArgsError::NonPositiveTop);
        }
        Ok(Self {
            temperature,
            top_p: f32::min(top_p, 1.),
            top_k,
        })
    }

    #[inline]
    pub fn is_argmax(&self) -> bool {
        self.temperature == 0.0 || self.top_k == 1
    }
}

