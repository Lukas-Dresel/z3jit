use inkwell::builder::BuilderError;


#[derive(Debug)]
pub enum Z3JitError {
    BuilderError(BuilderError),
    Other(String),
}

impl From<BuilderError> for Z3JitError {
    fn from(e: BuilderError) -> Self {
        Z3JitError::BuilderError(e)
    }
}
impl From<String> for Z3JitError {
    fn from(e: String) -> Self {
        Z3JitError::Other(e)
    }
}
impl From<&str> for Z3JitError {
    fn from(e: &str) -> Self {
        Z3JitError::Other(String::from(e))
    }
}