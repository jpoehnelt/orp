use ort::session::{SessionInputs, SessionOutputs};
use super::model::Model;
use super::Result;


/// Defines a generic pipeline
pub trait Pipeline<'a> {
    type Input;
    type Output;
    type Context;
    type Parameters;
    
    fn pre_processor(&self, params: &Self::Parameters) -> impl PreProcessor<'a, Self::Input, Self::Context>;
    
    fn post_processor(&self, params: &Self::Parameters) -> impl PostProcessor<Self::Output, Self::Context>;

    /// Optionally, the pipeline can expose the (exact) set of input tensors that must be exposed by the model 
    /// In such case it will be checked before inferencing.
    fn expected_inputs(&self, _params: &Self::Parameters) -> Option<impl Iterator<Item = &str>> {
        None::<std::iter::Empty<&str>>
    }

    /// Optionally, the pipeline can expose the (sub-)set of output tensors that must be exposed by the model
    /// In such case it will be checked before inferencing.
    fn expected_outputs(&self, _params: &Self::Parameters) -> Option<impl Iterator<Item = &str>> {
        None::<std::iter::Empty<&str>>
    }
}


/// Defines a generic pre-processor
pub trait PreProcessor<'a, I, C>: composable::Composable<I, (SessionInputs<'a, 'a>, C)> {}
impl<'a, I, C, T: composable::Composable<I, (SessionInputs<'a, 'a>, C)>> PreProcessor<'a, I, C> for T {}


/// Defines a generic post-processor
pub trait PostProcessor<O, C>: composable::Composable<(SessionOutputs<'static>, C), O> {}
impl<O, C, T: composable::Composable<(SessionOutputs<'static>, C), O>> PostProcessor<O, C> for T {}
